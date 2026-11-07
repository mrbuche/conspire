//! What the library has in place of the standard library, when it is built without it.
//!
//! The standard library is then not there, so the names that it has without asking come from
//! here, and the functions of floating-point numbers that `core` does not have as methods are in
//! [`FloatMath`](float::FloatMath).

mod float;

/// The names that every file of the library has, as `std` gives them.
///
/// The compiler does not see the use of a prelude, so it reports these as unused.
#[allow(unused_imports)]
pub(crate) mod prelude {
    pub(crate) use super::float::FloatMath;
    pub use alloc::{
        borrow::ToOwned,
        boxed::Box,
        string::{String, ToString},
        vec::Vec,
    };
    pub use core::prelude::rust_2024::*;
    /// The tests have the standard library, whose macros the prelude then has to provide.
    #[cfg(test)]
    pub use std::{dbg, eprint, eprintln, print, println};
}
