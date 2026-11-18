use crate::fem::element::solid::{
    hyperelastic::autodiff::test::elastic_tests, hyperviscoelastic::autodiff::test::viscous_tests,
};

elastic_tests!(crate::fem::element::quadratic::Hexahedron, 27, 27);
viscous_tests!(crate::fem::element::quadratic::Hexahedron, 27, 27);
