#[cfg(test)]
mod test;

use super::{GpuError, Result};
use core::{ffi::c_void, mem::size_of};

#[cfg(target_os = "linux")]
use {
    alloc::ffi::CString,
    core::{
        ffi::{c_char, c_int},
        mem::transmute_copy,
    },
};

#[cfg(target_os = "linux")]
#[cfg_attr(target_os = "linux", link(name = "dl"))]
unsafe extern "C" {
    fn dlopen(filename: *const c_char, flags: c_int) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
}

#[cfg(target_os = "linux")]
const RTLD_NOW: c_int = 2;

/// An open shared library, which stays open: the function pointers taken from it must outlive
/// everything that calls them, so it is never closed.
pub(super) struct Library(*mut c_void);

impl Library {
    /// The first of the libraries of the names that can be opened.
    pub(super) fn open(names: &[&str]) -> Result<Self> {
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
    pub(super) fn function<F: Copy>(&self, name: &str) -> Result<F> {
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
                Ok(unsafe { transmute_copy(&address) })
            }
        }
        #[cfg(not(target_os = "linux"))]
        {
            let _ = (&self.0, name);
            Err(GpuError::Unavailable("only Linux is supported".to_string()))
        }
    }
}
