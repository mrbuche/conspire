use core::{ffi::c_void, ptr::from_ref};

/// A value that a kernel takes as an argument.
///
/// # Safety
///
/// `pointer` must point at a value of the type that the kernel takes for that argument, which is
/// valid to read while the argument is borrowed.
pub unsafe trait Argument {
    /// A pointer to the value, which the launch reads.
    fn pointer(&self) -> *mut c_void;
}

macro_rules! scalar_argument {
    ($($type:ty),*) => {$(
        // SAFETY: the pointer is that of the value, which is borrowed, and a kernel that takes
        // this type takes a value of it.
        unsafe impl Argument for $type {
            fn pointer(&self) -> *mut c_void {
                from_ref(self).cast_mut().cast()
            }
        }
    )*};
}

scalar_argument!(u32, f64);
