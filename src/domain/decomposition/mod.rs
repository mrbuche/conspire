//! Domain decomposition methods.

pub use crate::geometry::mesh::{Bisection, Decomposition, DecompositionQuality};

#[cfg(feature = "fem")]
pub mod feti;
