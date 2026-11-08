//! A CUDA device, and the kernels of the library to run on it.
//!
//! The driver is opened when a device is made, not linked, so that a build needs no CUDA, and a
//! machine without it gets [`GpuError::Unavailable`] instead of a program that does not start.

mod block;
mod cuda;
mod kernels;
#[cfg(feature = "fem")]
mod resident;
#[cfg(test)]
mod test;

pub use block::{DeviceBlock, DeviceBlocks, DeviceLayout};
pub use cuda::{Argument, Buffer, Device, Function, Module, Resources};
#[cfg(feature = "fem")]
pub use resident::{DeviceError, Resident};

use core::{
    error::Error,
    fmt::{self, Debug, Display, Formatter},
};

/// Possible errors encountered on a device.
pub enum GpuError {
    /// There is no CUDA, or no such device.
    Unavailable(String),
    /// The kernels of the library were not built, so there is nothing to run on the device.
    KernelsNotBuilt,
    /// The kernels have none of this name, so the model or the shape of the elements of a block
    /// cannot be run on the device.
    NoKernel(String),
    /// The device found that an element of a block fails, and the host does not.
    Disagreement {
        /// The block, in the order of the model.
        block: usize,
        /// The element of the block.
        item: usize,
        /// The integration point of the element.
        point: usize,
    },
    /// The driver reported an error.
    Cuda(String),
}

impl Display for GpuError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unavailable(message) => write!(f, "No CUDA device is available: {message}."),
            Self::KernelsNotBuilt => write!(
                f,
                "The kernels for the device are not built: run kernels/build.sh."
            ),
            Self::NoKernel(name) => write!(
                f,
                "The kernels have no {name}: the model or the shape of the elements cannot run on a device."
            ),
            Self::Disagreement { block, item, point } => write!(
                f,
                "The device found that integration point {point} of element {item} of block {block} fails, and the host does not."
            ),
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
