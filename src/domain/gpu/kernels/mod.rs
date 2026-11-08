//! The PTX of the kernels, as it is given to the driver.

#[cfg(test)]
mod test;

use super::GpuError;
use alloc::borrow::Cow;

const BUILT: &str = include_str!(concat!(env!("OUT_DIR"), "/kernels.ptx"));

/// The name of the kernel for the elements of a block of a model, with the given number of
/// integration points and nodes, which `kernels/lib.rs` makes for every model and shape.
pub(super) fn block_forces(model: &str, points: usize, nodes: usize) -> String {
    format!("block_forces_{model}_g{points}_n{nodes}")
}

/// The PTX of the kernels of the library, with the multiplications, additions and subtractions
/// fused if `fused`.
///
/// The compiler writes them with the rounding mode named, `mul.rn.f64`, and the driver does not
/// fuse an instruction whose rounding is named into a fused multiply-add, which is two operations
/// in one for the arithmetic of a device. With `fused` the rounding is left unnamed, which lets
/// the driver fuse them, as it does for kernels written for the device. That takes the kernels
/// that make the gradient vectors as they go about 15% less time on an H100. The results differ
/// in the last bits, as they do when a fused multiply-add is used on any processor, so without
/// `fused` the rounding stays named, for the results that the operations alone give.
pub(super) fn ptx(fused: bool) -> Result<Cow<'static, str>, GpuError> {
    prepare(BUILT, fused)
}

fn prepare(ptx: &str, fused: bool) -> Result<Cow<'_, str>, GpuError> {
    if ptx.trim().is_empty() {
        Err(GpuError::KernelsNotBuilt)
    } else if fused {
        Ok(Cow::Owned(
            ptx.replace("mul.rn.f64", "mul.f64")
                .replace("add.rn.f64", "add.f64")
                .replace("sub.rn.f64", "sub.f64"),
        ))
    } else {
        Ok(Cow::Borrowed(ptx))
    }
}
