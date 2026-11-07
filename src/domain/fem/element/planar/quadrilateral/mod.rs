#[cfg(feature = "autodiff")]
mod autodiff;

use crate::fem::element::Element;

pub type Quadrilateral = Element<2, 4, 4, 1>;
