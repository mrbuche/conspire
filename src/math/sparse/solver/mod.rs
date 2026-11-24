#[cfg(test)]
mod test;

use super::{
    SparseError,
    factor::{CscLdl, CscLu},
    matrix::CscMatrix,
};
use crate::math::{Scalar, Vector};
use alloc::rc::Rc;
use core::cell::{Ref, RefCell};

/// Where the solves of a [`SparseSolver`] are done, in place of its own factorizations, such as on
/// a device.
///
/// The source gives the value at each position of the pattern the solver was made from, and
/// duplicate positions are summed by the backend.
pub trait SparseBackend {
    /// Solves for the values a source gives and a right-hand side.
    fn solve(
        &mut self,
        source: &mut dyn FnMut(usize, usize) -> Scalar,
        b: &Vector,
    ) -> Result<Vector, SparseError>;
}

/// A sparse direct solver for repeated solves on a fixed sparsity pattern.
///
/// Cloning shares the cached factorization, whose pivot order and fill pattern
/// are reused across solves. Symmetric values use an LDLᵀ factorization,
/// falling back to LU for that solve alone when a pivot degrades, and
/// otherwise. Symmetry is a caller-supplied guarantee about the source values,
/// not detected at runtime — a source whose symmetry can vary between solves
/// (e.g. a tangent that is only symmetric near a particular configuration)
/// must be declared asymmetric.
///
/// With a [`SparseBackend`] the solves are the backend's, which the clones share as they share the
/// factorizations; the inertia of [`SparseSolver::solve_ldl`] and the test of
/// [`SparseSolver::is_minimum`] remain those of the factorizations here.
#[derive(Clone)]
pub struct SparseSolver {
    symmetric: bool,
    backend: Option<Rc<RefCell<dyn SparseBackend>>>,
    matrix: RefCell<CscMatrix>,
    ldl: Rc<RefCell<Option<CscLdl>>>,
    lu: Rc<RefCell<Option<CscLu>>>,
}

impl SparseSolver {
    pub fn from_pattern(num: usize, pattern: Vec<(usize, usize)>, symmetric: bool) -> Self {
        let matrix = CscMatrix::from_pattern(num, num, pattern);
        let ldl = if symmetric {
            matrix.ldl_symbolic().ok()
        } else {
            None
        };
        let lu = if ldl.is_none() {
            matrix.lu_symbolic().ok()
        } else {
            None
        };
        Self {
            symmetric,
            backend: None,
            matrix: RefCell::new(matrix),
            ldl: Rc::new(RefCell::new(ldl)),
            lu: Rc::new(RefCell::new(lu)),
        }
    }
    /// The same solver, doing its solves with a backend, which is made for the same pattern.
    pub fn with_backend(mut self, backend: impl SparseBackend + 'static) -> Self {
        self.backend = Some(Rc::new(RefCell::new(backend)));
        self
    }
    /// The number of unknowns.
    pub fn size(&self) -> usize {
        self.matrix.borrow().height()
    }
    /// Whether the values were declared symmetric.
    pub fn is_symmetric(&self) -> bool {
        self.symmetric
    }
    /// The nonzero (row, column) positions this structure was built from.
    pub fn pattern(&self) -> Ref<'_, [(usize, usize)]> {
        Ref::map(self.matrix.borrow(), |matrix| matrix.pattern())
    }
    /// Solve using the LDLᵀ factorization alone, reporting its inertia and
    /// never substituting LU.
    ///
    /// A caller that gates on inertia — a trust region asking whether a shift
    /// has made the model admissible — needs this factorization's own verdict,
    /// which an LU rescue would quietly launder into an answer carrying no
    /// inertia at all. Errors instead: `Unsymmetric` if the solver was not
    /// built for symmetric values, or whatever the refactorization reports.
    pub fn solve_ldl(
        &self,
        source: impl FnMut(usize, usize) -> Scalar,
        b: &Vector,
    ) -> Result<(Vector, (usize, usize, usize)), SparseError> {
        let mut matrix = self.matrix.borrow_mut();
        matrix.fill(source);
        let mut ldl = self.ldl.borrow_mut();
        let cached = ldl.as_mut().ok_or(SparseError::Unsymmetric)?;
        cached.refactor(&matrix)?;
        Ok((cached.solve(b), cached.inertia()))
    }
    /// Whether the matrix of a source, the Hessian bordered by the given number of
    /// independent constraints, is positive definite on the constraints' null space.
    ///
    /// `None` when the LDLᵀ factorization cannot say, being unavailable for an
    /// asymmetric solver or broken down at these values: unverified is not a minimum.
    pub fn is_minimum(
        &self,
        source: impl FnMut(usize, usize) -> Scalar,
        constraints: usize,
    ) -> Option<bool> {
        let mut matrix = self.matrix.borrow_mut();
        matrix.fill(source);
        let mut ldl = self.ldl.borrow_mut();
        let cached = ldl.as_mut()?;
        cached.refactor(&matrix).ok()?;
        Some(cached.is_minimum(constraints))
    }
    /// Solve a system of linear equations with values from a source,
    /// refactoring the cached factorization when possible.
    ///
    /// A symmetric solver uses its LDLᵀ factorization, solving by LU for any
    /// solve in which a pivot degrades; LU carries no inertia, so see
    /// [`SparseSolver::solve_ldl`] for a solve that reports it instead of
    /// falling back.
    pub fn solve(
        &self,
        source: impl FnMut(usize, usize) -> Scalar,
        b: &Vector,
    ) -> Result<Vector, SparseError> {
        if let Some(backend) = &self.backend {
            let mut source = source;
            return backend.borrow_mut().solve(&mut source, b);
        }
        let mut matrix = self.matrix.borrow_mut();
        matrix.fill(source);
        let mut ldl = self.ldl.borrow_mut();
        if let Some(cached) = ldl.as_mut()
            && cached.refactor(&matrix).is_ok()
        {
            return Ok(cached.solve(b));
        }
        drop(ldl);
        let mut lu = self.lu.borrow_mut();
        if let Some(cached) = lu.as_mut()
            && cached.refactor(&matrix).is_ok()
        {
            return Ok(cached.solve(b));
        }
        let fresh = match matrix.lu_symbolic() {
            Ok(mut symbolic) => match symbolic.refactor(&matrix) {
                Ok(()) => symbolic,
                Err(_) => matrix.lu_amd()?,
            },
            Err(_) => matrix.lu_amd()?,
        };
        let solution = fresh.solve(b);
        *lu = Some(fresh);
        Ok(solution)
    }
}
