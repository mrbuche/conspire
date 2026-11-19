use super::{DevicePointer, GpuError, Result, library::Library};
use core::{
    ffi::{CStr, c_char, c_int, c_uint, c_void},
    ptr::null,
};

pub(super) struct Api {
    _driver: Library,
    pub(super) init: unsafe extern "C" fn(c_uint) -> c_int,
    pub(super) device_count: unsafe extern "C" fn(*mut c_int) -> c_int,
    pub(super) device_name: unsafe extern "C" fn(*mut c_char, c_int, c_int) -> c_int,
    pub(super) device_attribute: unsafe extern "C" fn(*mut c_int, c_int, c_int) -> c_int,
    pub(super) context_retain: unsafe extern "C" fn(*mut *mut c_void, c_int) -> c_int,
    pub(super) context_release: unsafe extern "C" fn(c_int) -> c_int,
    pub(super) context_set: unsafe extern "C" fn(*mut c_void) -> c_int,
    pub(super) synchronize: unsafe extern "C" fn() -> c_int,
    pub(super) alloc: unsafe extern "C" fn(*mut DevicePointer, usize) -> c_int,
    pub(super) free: unsafe extern "C" fn(DevicePointer) -> c_int,
    pub(super) upload: unsafe extern "C" fn(DevicePointer, *const c_void, usize) -> c_int,
    pub(super) download: unsafe extern "C" fn(*mut c_void, DevicePointer, usize) -> c_int,
    error_string: unsafe extern "C" fn(c_int, *mut *const c_char) -> c_int,
}

// SAFETY: the library is never closed and the functions are plain function pointers,
// which the driver documents as callable from any thread.
unsafe impl Send for Api {}

// SAFETY: as for `Send`, the functions can be called from any thread.
unsafe impl Sync for Api {}

impl Api {
    pub(super) fn load() -> Result<Self> {
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
            error_string: driver.function("cuGetErrorString")?,
            _driver: driver,
        })
    }
    /// Turns the status of a driver call into a result.
    pub(super) fn check(&self, status: c_int, call: &str) -> Result<()> {
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
