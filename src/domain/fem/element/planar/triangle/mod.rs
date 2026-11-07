#[cfg(feature = "autodiff")]
mod autodiff;

use crate::fem::element::Element;

pub type Triangle = Element<2, 1, 3, 1>;
