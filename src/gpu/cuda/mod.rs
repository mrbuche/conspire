//! The little of the CUDA driver that running kernels needs.
//!
//! The driver is opened when a [`Device`] is made rather than linked, so that a build needs no
//! CUDA, and a machine without it gets [`GpuError::Unavailable`] instead of a binary that does not
//! start. That is done for Linux; elsewhere there is no device to open.
//!
//! Everything that is made on a device keeps its context alive, and makes the context current on
//! the thread that uses it, so that it does not matter where it is dropped or used from.

mod api;
mod buffer;
mod context;
mod device;
mod library;

pub use buffer::Buffer;
pub use device::Device;

use super::GpuError;
use core::result;

type Result<T> = result::Result<T, GpuError>;
type DevicePointer = u64;
