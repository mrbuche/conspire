#[cfg(test)]
mod test;

use super::GpuError;
use alloc::borrow::Cow;
use core::fmt::{self, Display, Formatter};

/// The PTX of an application, which is embedded if the build script found it, as the directory of
/// the kernels says, and is empty if it is not built, which is an error that says which to build.
macro_rules! embed {
    ($constant:ident, $application:literal) => {
        #[cfg(conspire_ptx = $application)]
        const $constant: &str = include_str!(concat!($application, ".ptx"));
        #[cfg(not(conspire_ptx = $application))]
        const $constant: &str = "";
    };
}

embed!(ELEMENTS, "elements");

/// A set of kernels, which is built and loaded on its own.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Application {
    /// The kernels of the elements.
    Elements,
}

impl Application {
    /// The name of the directory of the kernels, which is that of their PTX.
    pub const fn name(self) -> &'static str {
        match self {
            Self::Elements => "elements",
        }
    }
    const fn built(self) -> &'static str {
        match self {
            Self::Elements => ELEMENTS,
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
