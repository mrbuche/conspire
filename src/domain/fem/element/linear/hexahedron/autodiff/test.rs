use crate::fem::element::solid::{
    hyperelastic::autodiff::test::elastic_tests, hyperviscoelastic::autodiff::test::viscous_tests,
};

elastic_tests!(crate::fem::element::linear::Hexahedron, 8, 8);
viscous_tests!(crate::fem::element::linear::Hexahedron, 8, 8);
