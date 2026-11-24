#[cfg(test)]
mod test;

mod cuda;
pub(crate) mod kernels;

#[cfg(feature = "elements")]
pub use crate::domain::block::device::{DeviceBlock, DeviceBlocks, DeviceLayout};
#[cfg(feature = "fem")]
pub use crate::domain::{DeviceError, Resident};
pub use cuda::{Argument, Buffer, Device, Function, Module, Resources};
pub use kernels::Application;

use core::{
    error::Error,
    fmt::{self, Debug, Display, Formatter},
};

/// Possible errors encountered on a device.
pub enum GpuError {
    /// There is no CUDA, or no such device.
    Unavailable(String),
    /// The kernels of an application were not built, so there is nothing to run on the device.
    KernelsNotBuilt(Application),
    /// The kernels have none of this name, so it cannot be run on the device.
    NoKernel(String),
    /// The device found an error at an element of a block, and the host does not.
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
            Self::KernelsNotBuilt(application) => write!(
                f,
                "The kernels of the {application} are not built: run kernels/build.sh {application}."
            ),
            Self::NoKernel(name) => write!(
                f,
                "The kernels have no {name}: the model or the shape of the elements cannot run on a device."
            ),
            Self::Disagreement { block, item, point } => write!(
                f,
                "The device found an error at integration point {point} of element {item} of block {block}, and the host does not."
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
