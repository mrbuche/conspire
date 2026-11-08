//! Domain discretization methods.

#[cfg(all(test, feature = "vem"))]
mod test;

pub(crate) mod block;
mod error;
pub use error::{AtItems, ElementModelError, ForEachAtItems, ItemError, Locate, Source};
#[cfg(feature = "cbm")]
pub mod cbm;
#[cfg(any(feature = "elements", feature = "fem"))]
pub mod fem;
#[cfg(feature = "fem")]
#[path = "decomposition/feti/mod.rs"]
pub mod feti;
#[cfg(feature = "gpu")]
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) mod gpu;
pub(crate) mod time_scale;
#[cfg(feature = "vem")]
pub mod vem;

#[cfg(any(feature = "cbm", feature = "fem", feature = "vem"))]
mod model;
#[cfg(any(feature = "cbm", feature = "fem", feature = "vem"))]
pub use model::*;
