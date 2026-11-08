#![doc = include_str!("../README.md")]
#![cfg_attr(not(any(feature = "std", test)), no_std)]
#![cfg_attr(
    not(feature = "std"),
    feature(core_float_math, prelude_import),
    allow(internal_features)
)]
#![cfg_attr(
    all(not(feature = "std"), target_arch = "nvptx64"),
    feature(stdarch_nvptx)
)]
#![cfg_attr(feature = "autodiff", feature(autodiff))]
#![cfg_attr(feature = "nightly", feature(portable_simd))]
#![warn(
    clippy::alloc_instead_of_core,
    clippy::std_instead_of_alloc,
    clippy::std_instead_of_core
)]

#[macro_use]
extern crate alloc;

#[cfg(not(feature = "std"))]
mod no_std;

#[cfg(not(feature = "std"))]
#[prelude_import]
#[allow(unused_imports)]
use no_std::prelude::*;

#[cfg(feature = "cbm")]
#[doc(hidden)]
pub use domain::cbm;

#[cfg(any(feature = "io", feature = "math"))]
#[cfg_attr(not(feature = "math"), allow(dead_code))]
mod hash;

#[cfg(feature = "constitutive")]
pub mod constitutive;

#[cfg(any(
    feature = "cbm",
    feature = "elements",
    feature = "fem",
    feature = "vem"
))]
#[path = "domain/mod.rs"]
pub mod domain;

#[cfg(any(feature = "elements", feature = "fem"))]
#[doc(hidden)]
pub use domain::fem;

#[cfg(feature = "fem")]
#[doc(hidden)]
pub use domain::feti;

#[cfg(feature = "gpu")]
#[doc(hidden)]
pub use domain::gpu;

#[cfg(feature = "geometry")]
pub mod geometry;

#[cfg(feature = "io")]
pub mod io;

#[cfg(feature = "math")]
pub mod math;

#[cfg(feature = "mechanics")]
pub mod mechanics;

#[cfg(feature = "physics")]
pub mod physics;

#[cfg(feature = "math")]
pub mod units;

#[cfg(feature = "vem")]
#[doc(hidden)]
pub use domain::vem;

/// Absolute tolerance.
pub const ABS_TOL: f64 = 1e-12;

/// Relative tolerance.
pub const REL_TOL: f64 = 1e-12;

/// A perturbation.
pub const EPSILON: f64 = 1e-6;
