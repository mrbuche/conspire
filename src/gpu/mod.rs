//! A CUDA device.
//!
//! The driver is opened when a device is made, not linked, so that a build needs no CUDA, and a
//! machine without it gets [`GpuError::Unavailable`] instead of a program that does not start.

mod cuda;
#[cfg(test)]
mod test;

#[allow(unused_imports)]
pub(crate) use cuda::{Buffer, Device};

use core::{
    error::Error,
    fmt::{self, Debug, Display, Formatter},
};

/// Possible errors encountered on a device.
pub enum GpuError {
    /// There is no CUDA, or no such device.
    Unavailable(String),
    /// The driver reported an error.
    Cuda(String),
}

impl Display for GpuError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unavailable(message) => write!(f, "No CUDA device is available: {message}."),
            Self::Cuda(message) => write!(f, "CUDA: {message}"),
        }
    }
}

impl Debug for GpuError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "\n{self}\n")
    }
}

impl Error for GpuError {}
