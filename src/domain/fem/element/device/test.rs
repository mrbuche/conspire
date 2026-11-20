use super::DeviceElement;
use crate::{
    domain::block::device::DeviceKind,
    fem::element::linear::{Hexahedron, Tetrahedron},
};

#[test]
fn an_element_that_integrates_at_all_of_its_points_shares_what_it_integrates_with() {
    let (derivatives, weights) = <Hexahedron as DeviceElement<8, 8>>::shared();
    assert_eq!(derivatives.len(), 8 * 8 * 3);
    assert_eq!(weights.len(), 8);
    assert_eq!(<Hexahedron as DeviceElement<8, 8>>::KIND, DeviceKind::Full);
    assert_eq!(<Hexahedron as DeviceElement<8, 8>>::VALUES, 0);
}

#[test]
fn an_element_of_one_point_shares_the_derivatives_of_its_point() {
    let (derivatives, weights) = <Tetrahedron<1> as DeviceElement<1, 4>>::shared();
    assert_eq!(derivatives.len(), 4 * 3);
    assert_eq!(weights.len(), 1);
}
