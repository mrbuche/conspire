//! Domain discretization methods.

#[cfg(all(test, feature = "vem"))]
mod test;

pub(crate) mod block;
#[cfg(feature = "cbm")]
pub mod cbm;
pub mod decomposition;
mod error;
#[cfg(feature = "fem")]
pub mod fem;
mod model;
pub(crate) mod time_scale;
#[cfg(feature = "vem")]
pub mod vem;

pub use error::{AtItems, ElementModelError, ForEachAtItems, ItemError, Locate, Source};
pub use model::*;
