//! The little of the CUDA driver that running kernels needs.
//!
//! The driver is opened when a [`Device`] is made rather than linked, so that a build needs no
//! CUDA, and a machine without it gets [`GpuError::Unavailable`] instead of a binary that does not
//! start. That is done for Linux; elsewhere there is no device to open.
//!
//! Everything that is made on a device keeps its context alive, and makes the context current on
//! the thread that uses it, so that it does not matter where it is dropped or used from.

#[cfg(test)]
mod test;

use super::GpuError;
use alloc::{ffi::CString, sync::Arc};
use core::{
    ffi::{CStr, c_char, c_int, c_uint, c_void},
    marker::PhantomData,
    mem::size_of,
    ptr::{null, null_mut},
};

#[cfg(target_os = "linux")]
#[cfg_attr(target_os = "linux", link(name = "dl"))]
unsafe extern "C" {
    fn dlopen(filename: *const c_char, flags: c_int) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
}

#[cfg(target_os = "linux")]
const RTLD_NOW: c_int = 2;
const COMPUTE_CAPABILITY_MAJOR: c_int = 75;
const COMPUTE_CAPABILITY_MINOR: c_int = 76;
const JIT_MAX_REGISTERS: c_int = 0;
const MAX_THREADS_PER_MULTIPROCESSOR: c_int = 39;
const FUNCTION_LOCAL_SIZE_BYTES: c_int = 3;
const FUNCTION_NUM_REGISTERS: c_int = 4;

type Result<T> = core::result::Result<T, GpuError>;
type DevicePointer = u64;

/// An open shared library, which stays open: the function pointers taken from it must outlive
/// everything that calls them, so it is never closed.
struct Library(*mut c_void);

impl Library {
    /// The first of the libraries of the names that can be opened.
    fn open(names: &[&str]) -> Result<Self> {
        #[cfg(target_os = "linux")]
        {
            names
                .iter()
                .find_map(|name| {
                    let name = CString::new(*name).ok()?;
                    // SAFETY: `name` is a valid C string.
                    let handle = unsafe { dlopen(name.as_ptr(), RTLD_NOW) };
                    (!handle.is_null()).then_some(Self(handle))
                })
                .ok_or_else(|| GpuError::Unavailable(format!("none of {names:?} could be opened")))
        }
        #[cfg(not(target_os = "linux"))]
        {
            Err(GpuError::Unavailable(format!(
                "{names:?} cannot be opened, as only Linux is supported"
            )))
        }
    }
    /// A function of the library, of the function pointer type `F`.
    fn function<F: Copy>(&self, name: &str) -> Result<F> {
        assert_eq!(size_of::<F>(), size_of::<*mut c_void>());
        #[cfg(target_os = "linux")]
        {
            let symbol = CString::new(name).expect("a name without nul");
            // SAFETY: the handle is open and `symbol` is a valid C string.
            let address = unsafe { dlsym(self.0, symbol.as_ptr()) };
            if address.is_null() {
                Err(GpuError::Unavailable(format!("no {name} in the library")))
            } else {
                // SAFETY: the address is that of a function, and `F` is a function pointer, which
                // the caller chooses to be of the signature that the function has.
                Ok(unsafe { core::mem::transmute_copy(&address) })
            }
        }
        #[cfg(not(target_os = "linux"))]
        {
            let _ = (&self.0, name);
            Err(GpuError::Unavailable("only Linux is supported".to_string()))
        }
    }
}

/// The functions of the driver.
struct Api {
    _driver: Library,
    init: unsafe extern "C" fn(c_uint) -> c_int,
    device_count: unsafe extern "C" fn(*mut c_int) -> c_int,
    device_name: unsafe extern "C" fn(*mut c_char, c_int, c_int) -> c_int,
    device_attribute: unsafe extern "C" fn(*mut c_int, c_int, c_int) -> c_int,
    context_retain: unsafe extern "C" fn(*mut *mut c_void, c_int) -> c_int,
    context_release: unsafe extern "C" fn(c_int) -> c_int,
    context_set: unsafe extern "C" fn(*mut c_void) -> c_int,
    synchronize: unsafe extern "C" fn() -> c_int,
    alloc: unsafe extern "C" fn(*mut DevicePointer, usize) -> c_int,
    free: unsafe extern "C" fn(DevicePointer) -> c_int,
    upload: unsafe extern "C" fn(DevicePointer, *const c_void, usize) -> c_int,
    download: unsafe extern "C" fn(*mut c_void, DevicePointer, usize) -> c_int,
    fill: unsafe extern "C" fn(DevicePointer, c_uint, usize) -> c_int,
    module_load: unsafe extern "C" fn(*mut *mut c_void, *const c_void) -> c_int,
    module_load_options: unsafe extern "C" fn(
        *mut *mut c_void,
        *const c_void,
        c_uint,
        *mut c_int,
        *mut *mut c_void,
    ) -> c_int,
    module_unload: unsafe extern "C" fn(*mut c_void) -> c_int,
    module_function: unsafe extern "C" fn(*mut *mut c_void, *mut c_void, *const c_char) -> c_int,
    launch: unsafe extern "C" fn(
        *mut c_void,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        *mut c_void,
        *mut *mut c_void,
        *mut *mut c_void,
    ) -> c_int,
    function_attribute: unsafe extern "C" fn(*mut c_int, c_int, *mut c_void) -> c_int,
    occupancy: unsafe extern "C" fn(*mut c_int, *mut c_void, c_int, usize) -> c_int,
    error_string: unsafe extern "C" fn(c_int, *mut *const c_char) -> c_int,
}

// SAFETY: the library is never closed and the functions are plain function pointers, which the
// driver documents as callable from any thread.
unsafe impl Send for Api {}
unsafe impl Sync for Api {}

impl Api {
    fn load() -> Result<Self> {
        let driver = Library::open(&["libcuda.so.1", "libcuda.so"])?;
        Ok(Self {
            init: driver.function("cuInit")?,
            device_count: driver.function("cuDeviceGetCount")?,
            device_name: driver.function("cuDeviceGetName")?,
            device_attribute: driver.function("cuDeviceGetAttribute")?,
            context_retain: driver.function("cuDevicePrimaryCtxRetain")?,
            context_release: driver.function("cuDevicePrimaryCtxRelease_v2")?,
            context_set: driver.function("cuCtxSetCurrent")?,
            synchronize: driver.function("cuCtxSynchronize")?,
            alloc: driver.function("cuMemAlloc_v2")?,
            free: driver.function("cuMemFree_v2")?,
            upload: driver.function("cuMemcpyHtoD_v2")?,
            download: driver.function("cuMemcpyDtoH_v2")?,
            fill: driver.function("cuMemsetD32_v2")?,
            module_load: driver.function("cuModuleLoadData")?,
            module_load_options: driver.function("cuModuleLoadDataEx")?,
            module_unload: driver.function("cuModuleUnload")?,
            module_function: driver.function("cuModuleGetFunction")?,
            launch: driver.function("cuLaunchKernel")?,
            function_attribute: driver.function("cuFuncGetAttribute")?,
            occupancy: driver.function("cuOccupancyMaxActiveBlocksPerMultiprocessor")?,
            error_string: driver.function("cuGetErrorString")?,
            _driver: driver,
        })
    }
    /// Turns the status of a driver call into a result.
    fn check(&self, status: c_int, call: &str) -> Result<()> {
        if status == 0 {
            return Ok(());
        }
        let mut message: *const c_char = null();
        // SAFETY: `message` is written with a static string, or left null.
        unsafe { (self.error_string)(status, &mut message) };
        let message = if message.is_null() {
            "unknown error".to_string()
        } else {
            // SAFETY: the driver returns a static C string.
            unsafe { CStr::from_ptr(message) }
                .to_string_lossy()
                .into_owned()
        };
        Err(GpuError::Cuda(format!(
            "{call} failed with {status}: {message}"
        )))
    }
}

/// The context of a device, which is released when the last of what was made in it is dropped.
struct Context {
    api: Arc<Api>,
    ordinal: c_int,
    handle: *mut c_void,
}

// SAFETY: the handle is only used to make the context current on the thread that uses it, which
// the driver allows from any thread.
unsafe impl Send for Context {}
unsafe impl Sync for Context {}

impl Context {
    /// Makes the context current on the calling thread.
    fn bind(&self) -> Result<()> {
        // SAFETY: the context is held by this value.
        self.api.check(
            unsafe { (self.api.context_set)(self.handle) },
            "cuCtxSetCurrent",
        )
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        // SAFETY: the context was retained for this value.
        unsafe { (self.api.context_release)(self.ordinal) };
    }
}

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
        // SAFETY: `pointer` is written by the driver.
        self.context.api.check(
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
        self.module_from_ptx_with(ptx, None)
    }
    /// As [`module_from_ptx`](Self::module_from_ptx), with the most registers that a thread may
    /// have in the kernels that the driver compiles, if `max_registers` is given.
    ///
    /// A thread has at most 255, and a kernel with so many has few threads on a multiprocessor to
    /// hide the latency of its memory and arithmetic with, so fewer registers can make it faster
    /// though some of what it holds then goes to local memory, which is slow. Which is faster
    /// depends on the device.
    pub fn module_from_ptx_with(&self, ptx: &str, max_registers: Option<u32>) -> Result<Module> {
        self.context.bind()?;
        let image =
            CString::new(ptx).map_err(|_| GpuError::Cuda("the PTX has a nul in it".to_string()))?;
        let mut handle = null_mut();
        let api = &self.context.api;
        // SAFETY: the image is a C string of PTX, which the driver compiles and loads.
        let status = match max_registers {
            Some(limit) => unsafe {
                let mut options = [JIT_MAX_REGISTERS];
                let mut values = [limit as usize as *mut c_void];
                (api.module_load_options)(
                    &mut handle,
                    image.as_ptr().cast(),
                    1,
                    options.as_mut_ptr(),
                    values.as_mut_ptr(),
                )
            },
            None => unsafe { (api.module_load)(&mut handle, image.as_ptr().cast()) },
        };
        api.check(status, "cuModuleLoadData")?;
        Ok(Module(Arc::new(ModuleHandle {
            context: self.context.clone(),
            handle,
        })))
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
        // SAFETY: the pointers point at values that live through the call, and the kernel
        // takes the arguments that it is given.
        self.context.api.check(
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

/// A value that a kernel takes as an argument.
pub trait Argument {
    /// A pointer to the value, which the launch reads.
    fn pointer(&self) -> *mut c_void;
}

macro_rules! scalar_argument {
    ($($type:ty),*) => {$(
        impl Argument for $type {
            fn pointer(&self) -> *mut c_void {
                core::ptr::from_ref(self).cast_mut().cast()
            }
        }
    )*};
}

scalar_argument!(u32, f64);

/// Values on the device.
pub struct Buffer<T> {
    context: Arc<Context>,
    pointer: DevicePointer,
    len: usize,
    element: PhantomData<T>,
}

impl<T> Argument for Buffer<T> {
    fn pointer(&self) -> *mut c_void {
        core::ptr::from_ref(&self.pointer).cast_mut().cast()
    }
}

impl<T: Copy> Buffer<T> {
    /// Copies `values` to the device, which must be as many as the buffer holds.
    pub fn upload(&self, values: &[T]) -> Result<()> {
        assert_eq!(values.len(), self.len, "a buffer is uploaded in full");
        if values.is_empty() {
            return Ok(());
        }
        self.context.bind()?;
        // SAFETY: the buffer holds as many values as `values` does.
        self.context.api.check(
            unsafe {
                (self.context.api.upload)(
                    self.pointer,
                    values.as_ptr().cast(),
                    core::mem::size_of_val(values),
                )
            },
            "cuMemcpyHtoD",
        )
    }
    /// Copies the buffer from the device into `values`, which must be as many as it holds.
    pub fn download(&self, values: &mut [T]) -> Result<()> {
        assert_eq!(values.len(), self.len, "a buffer is downloaded in full");
        if values.is_empty() {
            return Ok(());
        }
        self.context.bind()?;
        // SAFETY: the buffer holds as many values as `values` does.
        self.context.api.check(
            unsafe {
                (self.context.api.download)(
                    values.as_mut_ptr().cast(),
                    self.pointer,
                    core::mem::size_of_val(values),
                )
            },
            "cuMemcpyDtoH",
        )
    }
}

impl<T> Buffer<T> {
    /// Sets every 32 bits of the buffer to `value`.
    pub fn fill(&self, value: u32) -> Result<()> {
        self.context.bind()?;
        let words = (self.len * size_of::<T>()).div_ceil(4).max(1);
        // SAFETY: the allocation is at least `words` of 32 bits, as it is at least one value.
        self.context.api.check(
            unsafe { (self.context.api.fill)(self.pointer, value, words) },
            "cuMemsetD32",
        )
    }
}

impl<T> Drop for Buffer<T> {
    fn drop(&mut self) {
        // There is nothing to do about a failure while dropping, and a buffer that the device
        // could not free is lost with its context.
        if self.context.bind().is_ok() {
            // SAFETY: the allocation is that of this buffer.
            unsafe { (self.context.api.free)(self.pointer) };
        }
    }
}

/// A loaded module, which is unloaded when it and the kernels from it are dropped.
struct ModuleHandle {
    context: Arc<Context>,
    handle: *mut c_void,
}

// SAFETY: the handle is only used through the driver, which allows it from any thread with the
// context current.
unsafe impl Send for ModuleHandle {}
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
    /// A kernel of the module.
    pub fn function(&self, name: &str) -> Result<Function> {
        let name = CString::new(name).expect("a name without nul");
        let context = &self.0.context;
        context.bind()?;
        let mut handle = null_mut();
        // SAFETY: the module is loaded and `name` is a valid C string.
        context.api.check(
            unsafe { (context.api.module_function)(&mut handle, self.0.handle, name.as_ptr()) },
            "cuModuleGetFunction",
        )?;
        Ok(Function {
            module: self.0.clone(),
            handle,
        })
    }
}

/// A kernel, which keeps the module that it is from loaded.
pub struct Function {
    module: Arc<ModuleHandle>,
    handle: *mut c_void,
}

/// What a kernel needs of a multiprocessor of the device, which limits how many of its threads
/// the device runs at once, and so how well it hides the latency of its memory and arithmetic.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Resources {
    /// The registers of each thread, of at most 255, as the driver compiled the kernel for the
    /// device.
    pub registers: u32,
    /// The bytes of local memory of each thread, which is the memory of the device, and slow.
    pub local_bytes: u32,
    /// The threads in a block that the figures below are for.
    pub threads_per_block: u32,
    /// How many of those blocks fit on a multiprocessor at once.
    pub blocks_per_multiprocessor: u32,
    /// The most threads that a multiprocessor has at once.
    pub threads_per_multiprocessor: u32,
}

impl Resources {
    /// The fraction of the threads of a multiprocessor that the kernel has at once.
    pub fn occupancy(&self) -> f64 {
        f64::from(self.blocks_per_multiprocessor * self.threads_per_block)
            / f64::from(self.threads_per_multiprocessor.max(1))
    }
}

impl Function {
    /// What the kernel needs, for blocks of `threads` threads.
    pub fn resources(&self, threads: usize) -> Result<Resources> {
        let context = &self.module.context;
        context.bind()?;
        let api = &context.api;
        let (mut registers, mut local, mut blocks, mut maximum) = (0, 0, 0, 0);
        // SAFETY: the function is one of a loaded module, and the attributes are those of the
        // driver; the pointers are to locals.
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
