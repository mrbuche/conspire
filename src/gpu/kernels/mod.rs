//! The PTX of the kernels, as it is given to the driver.

#[cfg(test)]
mod test;

use super::GpuError;
use alloc::borrow::Cow;
use core::fmt::{self, Display, Formatter};

/// A set of kernels, which is built and loaded on its own.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Application {
    /// The kernels of the steps of dynamics.
    Dynamics,
    /// The kernels of the elements.
    Elements,
}

impl Application {
    /// The name of the directory of the kernels, which is that of their PTX.
    pub const fn name(self) -> &'static str {
        match self {
            Self::Dynamics => "dynamics",
            Self::Elements => "elements",
        }
    }
    const fn built(self) -> &'static str {
        match self {
            Self::Dynamics => include_str!(concat!(env!("OUT_DIR"), "/dynamics.ptx")),
            Self::Elements => include_str!(concat!(env!("OUT_DIR"), "/elements.ptx")),
        }
    }
}

impl Display for Application {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}

/// The PTX of the kernels of an application, with the operations fused if `fused`.
pub(crate) fn ptx(application: Application, fused: bool) -> Result<Cow<'static, str>, GpuError> {
    prepare(application, application.built(), fused)
}

/// The name of the kernel for the elements of a block of a model.
#[cfg(feature = "elements")]
pub(crate) fn block_forces(model: &str, points: usize, nodes: usize) -> String {
    format!("block_forces_{model}_g{points}_n{nodes}")
}

/// The name of the kernel for the elements of a point with hourglass control of a block of a model.
#[cfg(feature = "elements")]
pub(crate) fn block_forces_stabilized(model: &str, nodes: usize) -> String {
    format!("block_forces_{model}_stabilized_n{nodes}")
}

fn prepare(application: Application, ptx: &str, fused: bool) -> Result<Cow<'_, str>, GpuError> {
    if ptx.trim().is_empty() {
        Err(GpuError::KernelsNotBuilt(application))
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
