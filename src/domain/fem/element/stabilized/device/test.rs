use super::StabilizedElement;
use crate::{domain::block::device::DeviceKind, fem::element::DeviceElement};

type Hexahedron = StabilizedElement<8>;

#[test]
fn an_element_of_one_point_is_a_stabilized_kind_with_the_values_that_it_has() {
    assert_eq!(
        <Hexahedron as DeviceElement<1, 8>>::KIND,
        DeviceKind::Stabilized
    );
    assert_eq!(<Hexahedron as DeviceElement<1, 8>>::VALUES, 89);
}

#[test]
fn an_element_shares_nothing_with_the_others() {
    let (derivatives, weights) = <Hexahedron as DeviceElement<1, 8>>::shared();
    assert!(derivatives.is_empty() && weights.is_empty());
}
