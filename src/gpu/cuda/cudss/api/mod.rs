use super::super::{GpuError, Result, library::Library};
use core::ffi::{c_int, c_void};

pub(super) const PHASE_ANALYSIS: c_int = 1 | 2;
pub(super) const PHASE_FACTORIZATION: c_int = 4;
pub(super) const PHASE_REFACTORIZATION: c_int = 8;
pub(super) const PHASE_SOLVE: c_int = 16 | 32 | 64 | 128 | 256 | 512;
pub(super) const MATRIX_GENERAL: c_int = 0;
pub(super) const MATRIX_SYMMETRIC: c_int = 1;
pub(super) const VIEW_FULL: c_int = 0;
pub(super) const VIEW_UPPER: c_int = 2;
pub(super) const BASE_ZERO: c_int = 0;
pub(super) const LAYOUT_COLUMN_MAJOR: c_int = 0;
pub(super) const TYPE_F64: c_int = 1;
pub(super) const TYPE_I32: c_int = 10;

pub(super) type Handle = *mut c_void;

pub(super) struct Api {
    _library: Library,
    pub(super) create: unsafe extern "C" fn(*mut Handle) -> c_int,
    pub(super) destroy: unsafe extern "C" fn(Handle) -> c_int,
    pub(super) config_create: unsafe extern "C" fn(*mut Handle) -> c_int,
    pub(super) config_destroy: unsafe extern "C" fn(Handle) -> c_int,
    pub(super) data_create: unsafe extern "C" fn(Handle, *mut Handle) -> c_int,
    pub(super) data_destroy: unsafe extern "C" fn(Handle, Handle) -> c_int,
    pub(super) matrix_dense:
        unsafe extern "C" fn(*mut Handle, i64, i64, i64, *mut c_void, c_int, c_int) -> c_int,
    #[allow(clippy::type_complexity)]
    pub(super) matrix_csr: unsafe extern "C" fn(
        *mut Handle,
        i64,
        i64,
        i64,
        *mut c_void,
        *mut c_void,
        *mut c_void,
        *mut c_void,
        c_int,
        c_int,
        c_int,
        c_int,
        c_int,
        c_int,
    ) -> c_int,
    pub(super) matrix_destroy: unsafe extern "C" fn(Handle) -> c_int,
    pub(super) execute:
        unsafe extern "C" fn(Handle, c_int, Handle, Handle, Handle, Handle, Handle) -> c_int,
}

// SAFETY: the library is never closed and the functions are plain function pointers.
unsafe impl Send for Api {}

// SAFETY: as for `Send`.
unsafe impl Sync for Api {}

impl Api {
    pub(super) fn load() -> Result<Self> {
        let library = Library::open(&["libcudss.so.1", "libcudss.so.0", "libcudss.so"])?;
        Ok(Self {
            create: library.function("cudssCreate")?,
            destroy: library.function("cudssDestroy")?,
            config_create: library.function("cudssConfigCreate")?,
            config_destroy: library.function("cudssConfigDestroy")?,
            data_create: library.function("cudssDataCreate")?,
            data_destroy: library.function("cudssDataDestroy")?,
            matrix_dense: library.function("cudssMatrixCreateDn")?,
            matrix_csr: library.function("cudssMatrixCreateCsr")?,
            matrix_destroy: library.function("cudssMatrixDestroy")?,
            execute: library.function("cudssExecute")?,
            _library: library,
        })
    }
    /// Turns the status of a cuDSS call into a result.
    pub(super) fn check(&self, status: c_int, call: &str) -> Result<()> {
        if status == 0 {
            Ok(())
        } else {
            Err(GpuError::Cuda(format!(
                "{call} failed with cuDSS status {status}"
            )))
        }
    }
}
