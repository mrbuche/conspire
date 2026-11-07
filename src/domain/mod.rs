//! Domain discretization methods.

#[cfg(all(test, feature = "vem"))]
mod test;

pub(crate) mod block;
mod error;
pub use error::{AtItems, ElementModelError, ForEachAtItems, ItemError, Locate, Source};
#[cfg(feature = "cbm")]
pub mod cbm;
#[cfg(feature = "fem")]
pub mod fem;
#[cfg(feature = "fem")]
#[path = "decomposition/feti/mod.rs"]
pub mod feti;
pub(crate) mod time_scale;
#[cfg(feature = "vem")]
pub mod vem;

mod model;
pub use model::*;
