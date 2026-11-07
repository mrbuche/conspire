//! What the library has in place of the standard library, when it is built without it.
//!
//! The standard library is then not there, so the names that it has without asking come from
//! here.

/// The names that every file of the library has, as `std` gives them.
///
/// The compiler does not see the use of a prelude, so it reports these as unused.
#[allow(unused_imports)]
pub(crate) mod prelude {
    pub use alloc::{
        borrow::ToOwned,
        boxed::Box,
        string::{String, ToString},
        vec::Vec,
    };
    pub use core::prelude::rust_2024::*;
}
