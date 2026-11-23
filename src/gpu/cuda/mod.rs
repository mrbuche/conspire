mod api;
mod argument;
mod buffer;
mod context;
mod device;
mod library;
mod module;

pub use argument::Argument;
pub use buffer::Buffer;
pub use device::Device;
pub use module::{Function, Module};

use super::GpuError;
use core::result;

type Result<T> = result::Result<T, GpuError>;
type DevicePointer = u64;
