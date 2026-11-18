use crate::fem::element::solid::{
    hyperelastic::autodiff::test::elastic_tests, hyperviscoelastic::autodiff::test::viscous_tests,
};

elastic_tests!(crate::fem::element::quadratic::Wedge, 18, 15);
viscous_tests!(crate::fem::element::quadratic::Wedge, 18, 15);
