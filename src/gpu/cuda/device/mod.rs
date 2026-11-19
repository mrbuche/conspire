use super::{GpuError, Result, api::Api, buffer::Buffer, context::Context};
use alloc::sync::Arc;
use core::{
    ffi::{CStr, c_char, c_int},
    marker::PhantomData,
    mem::size_of,
    ptr::null_mut,
};

const COMPUTE_CAPABILITY_MAJOR: c_int = 75;
const COMPUTE_CAPABILITY_MINOR: c_int = 76;

/// A CUDA device, on which buffers and kernels live.
pub struct Device {
    context: Arc<Context>,
    name: String,
    capability: (c_int, c_int),
}

impl Device {
    /// Opens the device of the given ordinal.
    pub fn new(ordinal: usize) -> Result<Self> {
        let api = Arc::new(Api::load()?);
        // SAFETY: the calls are those of the driver API, with pointers to locals.
        unsafe {
            api.check((api.init)(0), "cuInit")
                .map_err(|error| GpuError::Unavailable(error.to_string()))?;
            let mut count = 0;
            api.check((api.device_count)(&mut count), "cuDeviceGetCount")?;
            if ordinal >= count as usize {
                return Err(GpuError::Unavailable(format!(
                    "device {ordinal} was asked for and {count} are present"
                )));
            }
            let ordinal = ordinal as c_int;
            let mut name = [0 as c_char; 128];
            api.check(
                (api.device_name)(name.as_mut_ptr(), name.len() as c_int, ordinal),
                "cuDeviceGetName",
            )?;
            let name = CStr::from_ptr(name.as_ptr()).to_string_lossy().into_owned();
            let (mut major, mut minor) = (0, 0);
            api.check(
                (api.device_attribute)(&mut major, COMPUTE_CAPABILITY_MAJOR, ordinal),
                "cuDeviceGetAttribute",
            )?;
            api.check(
                (api.device_attribute)(&mut minor, COMPUTE_CAPABILITY_MINOR, ordinal),
                "cuDeviceGetAttribute",
            )?;
            let mut handle = null_mut();
            api.check(
                (api.context_retain)(&mut handle, ordinal),
                "cuDevicePrimaryCtxRetain",
            )?;
            Ok(Self {
                context: Arc::new(Context {
                    api,
                    ordinal,
                    handle,
                }),
                name,
                capability: (major, minor),
            })
        }
    }
    /// The name of the device.
    #[cfg(test)]
    pub fn name(&self) -> &str {
        &self.name
    }
    /// The compute capability of the device, as a major and a minor number.
    #[cfg(test)]
    pub fn capability(&self) -> (i32, i32) {
        self.capability
    }
    /// Waits for everything that was launched.
    pub fn synchronize(&self) -> Result<()> {
        self.context.bind()?;
        // SAFETY: nothing is passed.
        self.context.api.check(
            unsafe { (self.context.api.synchronize)() },
            "cuCtxSynchronize",
        )
    }
    /// A buffer of `len` values on the device.
    pub fn buffer<T: Copy>(&self, len: usize) -> Result<Buffer<T>> {
        self.context.bind()?;
        let mut pointer = 0;
        let bytes = len.max(1) * size_of::<T>();
        self.context.api.check(
            // SAFETY: `pointer` is written by the driver.
            unsafe { (self.context.api.alloc)(&mut pointer, bytes) },
            "cuMemAlloc",
        )?;
        Ok(Buffer {
            context: self.context.clone(),
            pointer,
            len,
            element: PhantomData,
        })
    }
    /// A buffer on the device with the given values.
    pub fn buffer_of<T: Copy>(&self, values: &[T]) -> Result<Buffer<T>> {
        let buffer = self.buffer(values.len())?;
        buffer.upload(values)?;
        Ok(buffer)
    }
}
