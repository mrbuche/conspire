//! Domain discretization methods.

#[cfg(all(test, feature = "vem"))]
mod test;

pub(crate) mod block;
#[cfg(feature = "cbm")]
pub mod cbm;
#[cfg(feature = "fem")]
pub mod decomposition;
mod error;
#[cfg(any(feature = "elements", feature = "fem"))]
pub mod fem;
#[cfg(any(feature = "cbm", feature = "fem", feature = "vem"))]
mod model;
pub(crate) mod time_scale;
#[cfg(feature = "vem")]
pub mod vem;

pub use block::element::time_scale::courant_friedrichs_lewy::point_stiffness::{
    GradientGram, ModelBound, PointStiffness,
};
pub use error::{AtItems, ElementModelError, ForEachAtItems, ItemError, Locate, Source};
#[cfg(any(feature = "cbm", feature = "fem", feature = "vem"))]
pub use model::*;
