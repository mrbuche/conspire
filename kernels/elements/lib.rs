#![no_std]
#![feature(abi_ptx, stdarch_nvptx)]

use core::{
    arch::nvptx::{_block_dim_x, _block_idx_x, _thread_idx_x, trap},
    panic::PanicInfo,
};

#[panic_handler]
fn panic(_: &PanicInfo) -> ! {
    // SAFETY: stops the thread.
    unsafe { trap() }
}

fn thread() -> usize {
    // SAFETY: these read registers of the launch.
    unsafe { (_block_idx_x() as usize) * (_block_dim_x() as usize) + _thread_idx_x() as usize }
}

/// # Safety
///
/// `output` must have `count` values.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn squares(output: *mut f64, count: u32) {
    let index = thread();
    if index < count as usize {
        let value = (index + 1) as f64;
        // SAFETY: the index is within the values that `output` has.
        unsafe { *output.add(index) = value * value };
    }
}
