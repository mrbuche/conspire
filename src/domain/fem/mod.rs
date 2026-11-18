//! Finite element methods.

#[cfg(feature = "fem")]
pub mod block;
pub mod element;

#[cfg(feature = "fem")]
mod model;
#[cfg(feature = "fem")]
pub use model::*;
