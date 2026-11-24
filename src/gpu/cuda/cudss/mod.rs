#[cfg(test)]
mod test;

mod api;

use super::{Buffer, Device, GpuError, Result, context::Context};
use alloc::sync::Arc;
use api::{Api, Handle};
use core::{ffi::c_void, ptr::null_mut};

/// A direct solver of a sparse system of equations on a device, by cuDSS.
///
/// The matrix is in compressed sparse row form, with the pattern fixed for the life of the solver.
/// A symmetric matrix is given by its upper triangle alone. The ordering and the symbolic
/// factorization are made once, and each [`CudssSolver::solve`] refactors the new values.
pub struct CudssSolver {
    /// The functions of the cuDSS library.
    api: Api,
    /// The context of the device, made current before every call.
    context: Arc<Context>,
    /// The cuDSS library handle, from which the other handles are made.
    handle: Handle,
    /// The settings of the solver, which are cuDSS's defaults.
    config: Handle,
    /// What cuDSS keeps between phases: the ordering, the symbolic and numeric factors.
    data: Handle,
    /// The sparse matrix, which refers to `row_starts`, `columns` and `values`.
    matrix: Handle,
    /// The dense vector of unknowns, which refers to `unknowns`.
    solution: Handle,
    /// The dense vector of the right-hand side, which refers to `right_hand_sides`.
    right_hand_side: Handle,
    /// The values of the entries of the matrix, in the order of the pattern.
    values: Buffer<f64>,
    /// The solution, written by the solve phase.
    unknowns: Buffer<f64>,
    /// The right-hand side, uploaded before each solve.
    right_hand_sides: Buffer<f64>,
    /// The start of each row in `columns`, with one more at the end; kept alive for `matrix`.
    row_starts: Buffer<i32>,
    /// The column of each entry, zero-based; kept alive for `matrix`.
    columns: Buffer<i32>,
    /// The number of unknowns.
    size: usize,
    /// Whether the first factorization has been made, so that later solves refactor.
    factored: bool,
}

// SAFETY: the handles are only used through calls that bind the context first, which cuDSS
// allows from any one thread at a time.
unsafe impl Send for CudssSolver {}

impl CudssSolver {
    /// A solver for matrices of the pattern given by `row_starts` and `columns` (zero-based, with
    /// `size + 1` row starts), after the analysis of the pattern with `values` as the first guess.
    pub fn new(
        device: &Device,
        row_starts: &[i32],
        columns: &[i32],
        symmetric: bool,
        values: &[f64],
    ) -> Result<Self> {
        let size = row_starts.len().saturating_sub(1);
        assert_eq!(row_starts.first(), Some(&0), "row starts begin at zero");
        assert_eq!(
            row_starts[size] as usize,
            columns.len(),
            "row starts end at the number of entries"
        );
        assert_eq!(values.len(), columns.len(), "a value for each entry");
        let api = Api::load()?;
        let context = device.context().clone();
        let mut solver = Self {
            api,
            context,
            handle: null_mut(),
            config: null_mut(),
            data: null_mut(),
            matrix: null_mut(),
            solution: null_mut(),
            right_hand_side: null_mut(),
            values: device.buffer_of(values)?,
            unknowns: device.buffer::<f64>(size)?,
            right_hand_sides: device.buffer::<f64>(size)?,
            row_starts: device.buffer_of(row_starts)?,
            columns: device.buffer_of(columns)?,
            size,
            factored: false,
        };
        solver.context.bind()?;
        let (mtype, view) = if symmetric {
            (api::MATRIX_SYMMETRIC, api::VIEW_UPPER)
        } else {
            (api::MATRIX_GENERAL, api::VIEW_FULL)
        };
        let api = &solver.api;
        // SAFETY: the handles are written by the library, the buffers live as long as the solver
        // and hold what the descriptions say, and a failure leaves the handles to `drop`.
        unsafe {
            api.check((api.create)(&mut solver.handle), "cudssCreate")?;
            api.check((api.config_create)(&mut solver.config), "cudssConfigCreate")?;
            api.check(
                (api.data_create)(solver.handle, &mut solver.data),
                "cudssDataCreate",
            )?;
            api.check(
                (api.matrix_csr)(
                    &mut solver.matrix,
                    size as i64,
                    size as i64,
                    columns.len() as i64,
                    solver.row_starts.pointer as *mut c_void,
                    null_mut(),
                    solver.columns.pointer as *mut c_void,
                    solver.values.pointer as *mut c_void,
                    api::TYPE_I32,
                    api::TYPE_F64,
                    mtype,
                    view,
                    api::BASE_ZERO,
                ),
                "cudssMatrixCreateCsr",
            )?;
            api.check(
                (api.matrix_dense)(
                    &mut solver.solution,
                    size as i64,
                    1,
                    size as i64,
                    solver.unknowns.pointer as *mut c_void,
                    api::TYPE_F64,
                    api::LAYOUT_COLUMN_MAJOR,
                ),
                "cudssMatrixCreateDn",
            )?;
            api.check(
                (api.matrix_dense)(
                    &mut solver.right_hand_side,
                    size as i64,
                    1,
                    size as i64,
                    solver.right_hand_sides.pointer as *mut c_void,
                    api::TYPE_F64,
                    api::LAYOUT_COLUMN_MAJOR,
                ),
                "cudssMatrixCreateDn",
            )?;
        }
        solver.execute(api::PHASE_ANALYSIS, "analysis")?;
        Ok(solver)
    }
    /// The number of unknowns.
    pub fn size(&self) -> usize {
        self.size
    }
    /// Solves for the values of the entries, in the order of the pattern, and a right-hand side,
    /// factoring anew at the new values.
    pub fn solve(&mut self, values: &[f64], right_hand_side: &[f64]) -> Result<Vec<f64>> {
        assert_eq!(
            right_hand_side.len(),
            self.size,
            "a right-hand side entry per unknown"
        );
        self.values.upload(values)?;
        self.right_hand_sides.upload(right_hand_side)?;
        if self.factored {
            self.execute(api::PHASE_REFACTORIZATION, "refactorization")?;
        } else {
            self.execute(api::PHASE_FACTORIZATION, "factorization")?;
            self.factored = true;
        }
        self.execute(api::PHASE_SOLVE, "solve")?;
        let mut solution = vec![0.0; self.size];
        self.unknowns.download(&mut solution)?;
        Ok(solution)
    }
    fn execute(&self, phase: i32, name: &str) -> Result<()> {
        self.context.bind()?;
        let api = &self.api;
        // SAFETY: every handle was made by this solver and is alive.
        let status = unsafe {
            (api.execute)(
                self.handle,
                phase,
                self.config,
                self.data,
                self.matrix,
                self.solution,
                self.right_hand_side,
            )
        };
        api.check(status, &format!("cudssExecute ({name})"))?;
        let driver = &self.context.api;
        // SAFETY: nothing is passed.
        driver
            .check(unsafe { (driver.synchronize)() }, "cuCtxSynchronize")
            .map_err(|error| GpuError::Cuda(format!("after the {name}: {error}")))
    }
}

impl Drop for CudssSolver {
    fn drop(&mut self) {
        let _ = self.context.bind();
        let api = &self.api;
        // SAFETY: each handle that is not null was made by this solver, and is destroyed once.
        unsafe {
            for matrix in [self.matrix, self.solution, self.right_hand_side] {
                if !matrix.is_null() {
                    (api.matrix_destroy)(matrix);
                }
            }
            if !self.data.is_null() {
                (api.data_destroy)(self.handle, self.data);
            }
            if !self.config.is_null() {
                (api.config_destroy)(self.config);
            }
            if !self.handle.is_null() {
                (api.destroy)(self.handle);
            }
        }
    }
}
