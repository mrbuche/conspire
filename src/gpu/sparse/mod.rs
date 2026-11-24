#[cfg(test)]
mod test;

mod fill;
mod layout;

use super::{CudssSolver, Device, Timings};
use crate::math::{
    Execution, Scalar, Tensor, Vector,
    sparse::{SparseBackend, SparseError, SparseSolver},
};
use fill::{MINIMUM_PER_THREAD, fill};
use layout::Layout;
use std::time::Instant;

/// A sparse direct solver on a device, for repeated solves on a fixed sparsity pattern.
///
/// It is made as a [`SparseSolver`](crate::math::sparse::SparseSolver) is, from the pattern of the
/// nonzero positions, and solves for the values that a source gives, as that does: the pattern is
/// analyzed once, on the device, and each solve factors anew at the new values.
///
/// Symmetric values are factored as symmetric, by LDLᵀ with pivoting, and the pattern must then be
/// symmetric. Unlike a host solver there is no fallback to LU for a solve that breaks down: it is
/// an error.
///
/// The values are asked of the source at the positions of the device, and on the threads that the
/// [`Execution`] gives, which never changes them.
pub struct DeviceSparseSolver {
    /// The (row, column) that the source is asked for the value of each entry of the device.
    positions: Vec<(usize, usize)>,
    /// How many times each position is in the pattern, which its values are summed for.
    counts: Vec<u32>,
    /// The values the device has, in the order of its rows.
    values: Vec<Scalar>,
    /// How the values are computed.
    execution: Execution,
    /// The solver on the device, which a matrix of no unknowns does without.
    solver: Option<CudssSolver>,
    /// Where the time of the last solve went.
    timings: Timings,
}

impl DeviceSparseSolver {
    /// A solver for `num` unknowns and the nonzero (row, column) positions of `pattern`, whose
    /// duplicates are summed.
    ///
    /// Errors if `symmetric` is set and the pattern is not, or if a device does not do it.
    pub fn from_pattern(
        device: &Device,
        num: usize,
        pattern: Vec<(usize, usize)>,
        symmetric: bool,
    ) -> Result<Self, SparseError> {
        let Layout {
            row_starts,
            columns,
            positions,
            counts,
        } = Layout::new(num, pattern, symmetric)?;
        let solver = if num == 0 {
            None
        } else {
            Some(
                CudssSolver::new(device, &row_starts, &columns, symmetric)
                    .map_err(|error| SparseError::Device(error.to_string()))?,
            )
        };
        Ok(Self {
            values: vec![0.0; positions.len()],
            positions,
            counts,
            execution: Execution::Serial,
            solver,
            timings: Timings::default(),
        })
    }
    /// The same solver, computing the values of its matrix as the execution says.
    pub fn with_execution(mut self, execution: Execution) -> Self {
        self.execution = execution;
        self
    }
    /// Where the time of the last solve went.
    pub fn timings(&self) -> Timings {
        self.timings
    }
    /// Solves a system of linear equations with values from a source, which is asked for the value
    /// at each position of the pattern, and may be asked from several threads at once.
    pub fn solve(
        &mut self,
        source: impl Fn(usize, usize) -> Scalar + Sync,
        b: &Vector,
    ) -> Result<Vector, SparseError> {
        let Some(solver) = self.solver.as_mut() else {
            return Ok(Vector::zero(0));
        };
        let start = Instant::now();
        fill(
            &self.positions,
            &self.counts,
            &source,
            &mut self.values,
            self.execution.threads(),
            MINIMUM_PER_THREAD,
        );
        let right_hand_side = b.iter().copied().collect::<Vec<_>>();
        let fill = start.elapsed().as_secs_f64();
        let solution = solver
            .solve(&self.values, &right_hand_side)
            .map_err(|error| SparseError::Device(error.to_string()))?;
        self.timings = Timings {
            fill,
            ..solver.timings()
        };
        Ok(Vector::from(solution))
    }
}

/// The solver of a device that is made at the first solve, so that a pattern can be given to a
/// [`SparseSolver`] without the device being asked: whatever it cannot do is the error of the solve.
struct OnDevice {
    device: Device,
    size: usize,
    pattern: Vec<(usize, usize)>,
    symmetric: bool,
    execution: Execution,
    solver: Option<DeviceSparseSolver>,
}

impl SparseBackend for OnDevice {
    fn solve(
        &mut self,
        source: &(dyn Fn(usize, usize) -> Scalar + Sync),
        b: &Vector,
    ) -> Result<Vector, SparseError> {
        let solver = match &mut self.solver {
            Some(solver) => solver,
            none => {
                let made = DeviceSparseSolver::from_pattern(
                    &self.device,
                    self.size,
                    self.pattern.clone(),
                    self.symmetric,
                )?;
                self.pattern = Vec::new();
                none.insert(made.with_execution(self.execution))
            }
        };
        solver.solve(source, b)
    }
}

impl SparseSolver {
    /// The same solver, doing its solves on a device, by cuDSS, with the values of its matrix
    /// computed as the execution says.
    ///
    /// The device is first asked at the first solve, so that a device or a library that is not
    /// there, a pattern that is not symmetric when it is declared so, or a matrix the device cannot
    /// factor, is the error of a solve. The solves have no fallback to the host.
    pub fn on_device(self, device: &Device, execution: Execution) -> Self {
        let backend = OnDevice {
            device: device.clone(),
            size: self.size(),
            pattern: self.pattern().to_vec(),
            symmetric: self.is_symmetric(),
            execution,
            solver: None,
        };
        self.with_backend(backend)
    }
}
