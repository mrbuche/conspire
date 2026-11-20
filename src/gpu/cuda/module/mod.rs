use super::{Result, context::Context, resources::Resources};
use alloc::{ffi::CString, sync::Arc};
use core::{
    ffi::{c_int, c_void},
    ptr::null_mut,
};

const FUNCTION_LOCAL_SIZE_BYTES: c_int = 3;
const FUNCTION_NUM_REGISTERS: c_int = 4;
const MAX_THREADS_PER_MULTIPROCESSOR: c_int = 39;

/// A loaded module, which is unloaded when it and the kernels from it are dropped.
struct ModuleHandle {
    context: Arc<Context>,
    handle: *mut c_void,
}

// SAFETY: the handle is only used through the driver, which allows it from any thread with the
// context current.
unsafe impl Send for ModuleHandle {}

// SAFETY: as for `Send`, the handle is used from any thread.
unsafe impl Sync for ModuleHandle {}

impl Drop for ModuleHandle {
    fn drop(&mut self) {
        if self.context.bind().is_ok() {
            // SAFETY: the module is that loaded for this value.
            unsafe { (self.context.api.module_unload)(self.handle) };
        }
    }
}

/// Kernels that are loaded on the device.
pub struct Module(Arc<ModuleHandle>);

impl Module {
    /// The module that the driver loaded with the handle, in the context.
    pub(super) fn new(context: Arc<Context>, handle: *mut c_void) -> Self {
        Self(Arc::new(ModuleHandle { context, handle }))
    }
    /// A kernel of the module.
    pub fn function(&self, name: &str) -> Result<Function> {
        let name = CString::new(name).expect("a name without nul");
        let context = &self.0.context;
        context.bind()?;
        let mut handle = null_mut();
        context.api.check(
            // SAFETY: the module is loaded and `name` is a valid C string.
            unsafe { (context.api.module_function)(&mut handle, self.0.handle, name.as_ptr()) },
            "cuModuleGetFunction",
        )?;
        Ok(Function {
            _module: self.0.clone(),
            handle,
        })
    }
}

/// A kernel, which keeps the module that it is from loaded.
pub struct Function {
    _module: Arc<ModuleHandle>,
    pub(super) handle: *mut c_void,
}

impl Function {
    /// What the kernel needs of a multiprocessor, for blocks of `threads` threads.
    pub fn resources(&self, threads: usize) -> Result<Resources> {
        let context = &self._module.context;
        context.bind()?;
        let api = &context.api;
        let (mut registers, mut local, mut blocks, mut maximum) = (0, 0, 0, 0);
        // SAFETY: the function is one of a loaded module, the attributes are those of the
        // driver, and the pointers are to locals.
        unsafe {
            api.check(
                (api.function_attribute)(&mut registers, FUNCTION_NUM_REGISTERS, self.handle),
                "cuFuncGetAttribute",
            )?;
            api.check(
                (api.function_attribute)(&mut local, FUNCTION_LOCAL_SIZE_BYTES, self.handle),
                "cuFuncGetAttribute",
            )?;
            api.check(
                (api.occupancy)(&mut blocks, self.handle, threads as c_int, 0),
                "cuOccupancyMaxActiveBlocksPerMultiprocessor",
            )?;
            api.check(
                (api.device_attribute)(
                    &mut maximum,
                    MAX_THREADS_PER_MULTIPROCESSOR,
                    context.ordinal,
                ),
                "cuDeviceGetAttribute",
            )?;
        }
        Ok(Resources {
            registers: registers as u32,
            local_bytes: local as u32,
            threads_per_block: threads as u32,
            blocks_per_multiprocessor: blocks as u32,
            threads_per_multiprocessor: maximum as u32,
        })
    }
}
