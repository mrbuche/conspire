use super::{Result, api::Api};
use alloc::sync::Arc;
use core::ffi::{c_int, c_void};

/// The context of a device, which is released when the last of what was made in it is dropped.
pub(super) struct Context {
    pub(super) api: Arc<Api>,
    pub(super) ordinal: c_int,
    pub(super) handle: *mut c_void,
}

// SAFETY: the handle is only used to make the context current on the thread that uses it,
// which the driver allows from any thread.
unsafe impl Send for Context {}

// SAFETY: as for `Send`, the handle is used from any thread.
unsafe impl Sync for Context {}

impl Context {
    /// Makes the context current on the calling thread.
    pub(super) fn bind(&self) -> Result<()> {
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
