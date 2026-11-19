use super::{DevicePointer, Result, context::Context};
use alloc::sync::Arc;
use core::{marker::PhantomData, mem::size_of_val};

/// Values on the device.
pub struct Buffer<T> {
    pub(super) context: Arc<Context>,
    pub(super) pointer: DevicePointer,
    pub(super) len: usize,
    pub(super) element: PhantomData<T>,
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
                (self.context.api.upload)(self.pointer, values.as_ptr().cast(), size_of_val(values))
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
                    size_of_val(values),
                )
            },
            "cuMemcpyDtoH",
        )
    }
}

impl<T> Drop for Buffer<T> {
    fn drop(&mut self) {
        if self.context.bind().is_ok() {
            // SAFETY: the allocation is that of this buffer.
            unsafe { (self.context.api.free)(self.pointer) };
        }
    }
}
