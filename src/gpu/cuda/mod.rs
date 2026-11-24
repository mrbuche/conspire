mod api;
mod argument;
mod buffer;
mod context;
mod cudss;
mod device;
mod library;
mod module;
mod resources;

pub use argument::Argument;
pub use buffer::Buffer;
pub use cudss::CudssSolver;
pub use device::Device;
pub use module::{Function, Module};
pub use resources::Resources;

use super::GpuError;
use core::result;

type Result<T> = result::Result<T, GpuError>;
type DevicePointer = u64;
