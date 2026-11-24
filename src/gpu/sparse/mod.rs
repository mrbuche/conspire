#[cfg(test)]
mod test;

use super::{CudssSolver, Device};
use crate::math::{
    Scalar, Tensor, Vector,
    sparse::{CscMatrix, SparseBackend, SparseError, SparseSolver},
};

/// A sparse direct solver on a device, for repeated solves on a fixed sparsity pattern.
///
/// It is made as a [`SparseSolver`](crate::math::sparse::SparseSolver) is, from the pattern of the
/// nonzero positions, and solves for the values that a source gives, as that does: the pattern is
/// analyzed once, on the device, and each solve factors anew at the new values.
///
/// Symmetric values are factored as symmetric, by LDLᵀ with pivoting, and the pattern must then be
/// symmetric. Unlike a host solver there is no fallback to LU for a solve that breaks down: it is
/// an error.
pub struct DeviceSparseSolver {
    /// The matrix, in the order of the host: by column, by row within a column.
    matrix: CscMatrix,
    /// For each entry of the matrix the device has, the position of its value in `matrix`.
    gather: Vec<usize>,
    /// The values of `matrix`, in order, as one solve reads them.
    from_matrix: Vec<Scalar>,
    /// The values the device has, in the order of its rows.
    to_device: Vec<Scalar>,
    /// The solver on the device, which a matrix of no unknowns does without.
    solver: Option<CudssSolver>,
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
        let matrix = CscMatrix::from_pattern(num, num, pattern);
        if symmetric
            && matrix
                .iter()
                .any(|(i, j, _)| matrix.column(i).all(|(row, _)| row != j))
        {
            return Err(SparseError::Unsymmetric);
        }
        let mut entries = matrix
            .iter()
            .enumerate()
            .filter_map(|(position, (i, j, _))| {
                if !symmetric {
                    Some((i, j, position))
                } else {
                    (i >= j).then_some((j, i, position))
                }
            })
            .collect::<Vec<_>>();
        entries.sort_unstable();
        let too_large = || SparseError::Device("the matrix is too large for 32-bit indices".into());
        let mut row_starts = vec![0_i32; num + 1];
        entries
            .iter()
            .try_for_each(|&(row, _, _)| {
                row_starts[row + 1] = row_starts[row + 1].checked_add(1)?;
                Some(())
            })
            .ok_or_else(too_large)?;
        (0..num).for_each(|row| row_starts[row + 1] += row_starts[row]);
        let columns = entries
            .iter()
            .map(|&(_, column, _)| i32::try_from(column).map_err(|_| too_large()))
            .collect::<Result<Vec<_>, _>>()?;
        let gather = entries.iter().map(|&(_, _, position)| position).collect();
        let to_device = vec![0.0; entries.len()];
        let solver = if num == 0 {
            None
        } else {
            Some(
                CudssSolver::new(device, &row_starts, &columns, symmetric)
                    .map_err(|error| SparseError::Device(error.to_string()))?,
            )
        };
        Ok(Self {
            from_matrix: Vec::with_capacity(matrix.nonzeros()),
            matrix,
            gather,
            to_device,
            solver,
        })
    }
    /// The nonzero (row, column) positions this structure was built from.
    pub fn pattern(&self) -> &[(usize, usize)] {
        self.matrix.pattern()
    }
    /// Solves a system of linear equations with values from a source, which is asked for the value
    /// at each position of the pattern.
    pub fn solve(
        &mut self,
        source: impl FnMut(usize, usize) -> Scalar,
        b: &Vector,
    ) -> Result<Vector, SparseError> {
        let Some(solver) = self.solver.as_mut() else {
            return Ok(Vector::zero(0));
        };
        self.matrix.fill(source);
        self.from_matrix.clear();
        self.from_matrix
            .extend(self.matrix.iter().map(|(_, _, value)| *value));
        self.to_device
            .iter_mut()
            .zip(&self.gather)
            .for_each(|(value, &position)| *value = self.from_matrix[position]);
        let right_hand_side = b.iter().copied().collect::<Vec<_>>();
        solver
            .solve(&self.to_device, &right_hand_side)
            .map(Vector::from)
            .map_err(|error| SparseError::Device(error.to_string()))
    }
}

/// The solver of a device that is made at the first solve, so that a pattern can be given to a
/// [`SparseSolver`] without the device being asked: whatever it cannot do is the error of the solve.
struct OnDevice {
    device: Device,
    size: usize,
    pattern: Vec<(usize, usize)>,
    symmetric: bool,
    solver: Option<DeviceSparseSolver>,
}

impl SparseBackend for OnDevice {
    fn solve(
        &mut self,
        source: &mut dyn FnMut(usize, usize) -> Scalar,
        b: &Vector,
    ) -> Result<Vector, SparseError> {
        let solver = match &mut self.solver {
            Some(solver) => solver,
            none => none.insert(DeviceSparseSolver::from_pattern(
                &self.device,
                self.size,
                self.pattern.clone(),
                self.symmetric,
            )?),
        };
        solver.solve(source, b)
    }
}

impl SparseSolver {
    /// The same solver, doing its solves on a device, by cuDSS.
    ///
    /// The device is first asked at the first solve, so that a device or a library that is not
    /// there, a pattern that is not symmetric when it is declared so, or a matrix the device cannot
    /// factor, is the error of a solve. The solves have no fallback to the host.
    pub fn on_device(self, device: &Device) -> Self {
        let backend = OnDevice {
            device: device.clone(),
            size: self.size(),
            pattern: self.pattern().to_vec(),
            symmetric: self.is_symmetric(),
            solver: None,
        };
        self.with_backend(backend)
    }
}
