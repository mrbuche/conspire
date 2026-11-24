use super::{
    Argument, Function, GpuError, Module, Result, api::Api, buffer::Buffer, context::Context,
};
use alloc::{ffi::CString, sync::Arc};
use core::{
    ffi::{CStr, c_char, c_int, c_uint},
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
    pub fn name(&self) -> &str {
        &self.name
    }
    /// The compute capability of the device, as a major and a minor number.
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
    /// Loads kernels that are PTX, such as those that Rust code is built to, which the driver
    /// compiles for this device when they are loaded.
    pub fn module_from_ptx(&self, ptx: &str) -> Result<Module> {
        self.context.bind()?;
        let image =
            CString::new(ptx).map_err(|_| GpuError::Cuda("the PTX has a nul in it".to_string()))?;
        let mut handle = null_mut();
        self.context.api.check(
            // SAFETY: the image is a C string of PTX, which the driver compiles and loads.
            unsafe { (self.context.api.module_load)(&mut handle, image.as_ptr().cast()) },
            "cuModuleLoadData",
        )?;
        Ok(Module::new(self.context.clone(), handle))
    }
    /// Launches `threads` threads of a kernel in blocks of `block`, and returns without waiting.
    pub fn launch(
        &self,
        function: &Function,
        threads: usize,
        block: usize,
        arguments: &[&dyn Argument],
    ) -> Result<()> {
        if threads == 0 {
            return Ok(());
        }
        self.context.bind()?;
        let mut pointers = arguments
            .iter()
            .map(|argument| argument.pointer())
            .collect::<Vec<_>>();
        let grid = threads.div_ceil(block) as c_uint;
        self.context.api.check(
            // SAFETY: the pointers point at values that live through the call, and the kernel
            // takes the arguments that it is given.
            unsafe {
                (self.context.api.launch)(
                    function.handle,
                    grid,
                    1,
                    1,
                    block as c_uint,
                    1,
                    1,
                    0,
                    null_mut(),
                    pointers.as_mut_ptr(),
                    null_mut(),
                )
            },
            "cuLaunchKernel",
        )
    }
}
