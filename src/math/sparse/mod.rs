mod factor;
mod matrix;
mod solver;

pub use factor::{CscLdl, CscLu};
pub use matrix::CscMatrix;
pub use solver::SparseSolver;

use crate::math::assert::AssertionError;
use core::{error::Error, fmt};

/// Possible errors for sparse data types.
#[derive(PartialEq)]
pub enum SparseError {
    Singular,
    Unsymmetric,
    /// A device failed, or is not there, with what it reported.
    Device(String),
}

impl fmt::Display for SparseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Singular => write!(f, "Matrix is singular."),
            Self::Unsymmetric => write!(f, "Matrix is not symmetric."),
            Self::Device(message) => write!(f, "Device: {message}"),
        }
    }
}

impl fmt::Debug for SparseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "\n{self}\n")
    }
}

impl Error for SparseError {}

impl From<SparseError> for AssertionError {
    fn from(error: SparseError) -> Self {
        Self {
            message: error.to_string(),
        }
    }
}
