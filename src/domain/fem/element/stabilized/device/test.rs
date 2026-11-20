use super::StabilizedElement;
use crate::{
    domain::block::device::DeviceKind,
    fem::element::{DeviceElement, ElementNodalReferenceCoordinates, FiniteElement},
    math::assert::{Assert, AssertionError},
};

use core::array::from_fn;

type Hexahedron = StabilizedElement<8>;

fn bent() -> Hexahedron {
    let reference = <Hexahedron as FiniteElement<1, 3, 8, 8>>::parametric_reference();
    let nodes: [[f64; 3]; 8] = from_fn(|a| {
        let xi = &reference[a];
        let wobble = |i: usize| 0.05 * (1.3 * a as f64 + i as f64).sin();
        [
            1.1 * xi[0].value() + 0.2 * xi[1].value() + wobble(0),
            0.9 * xi[1].value() - 0.1 * xi[2].value() + wobble(1),
            1.2 * xi[2].value() + 0.1 * xi[0].value() + wobble(2),
        ]
    });
    ElementNodalReferenceCoordinates::<8>::from(nodes).into()
}

#[test]
fn an_element_has_as_many_values_as_it_says() {
    let element = bent();
    assert_eq!(element.values().len(), Hexahedron::VALUES);
    assert_eq!(Hexahedron::VALUES, 24 + 1 + 64);
    assert_eq!(<Hexahedron as DeviceElement<1, 8>>::VALUES, 89);
    assert_eq!(
        <Hexahedron as DeviceElement<1, 8>>::KIND,
        DeviceKind::Stabilized
    );
}

#[test]
fn an_element_shares_nothing_with_the_others() {
    let (derivatives, weights) = <Hexahedron as DeviceElement<1, 8>>::shared();
    assert!(derivatives.is_empty() && weights.is_empty());
}

#[test]
fn the_values_of_an_element_are_its_mean_gradient_its_volume_and_its_hourglass_matrix()
-> Result<(), AssertionError> {
    let element = bent();
    let values = element.values();
    for a in 0..8 {
        for k in 0..3 {
            Assert::eq(
                values[3 * a + k],
                element.gradient_vectors()[0][a][k].value(),
            )?;
        }
    }
    Assert::eq(values[24], element.integration_weights[0].value())?;
    for a in 0..8 {
        for b in 0..8 {
            Assert::eq(
                values[25 + 8 * a + b],
                element.hourglass_matrix()[a][b].value(),
            )?;
        }
    }
    Ok(())
}

#[test]
fn the_element_made_from_its_values_has_them_again_to_the_bit() {
    let element = bent();
    let values = element.values();
    let again = Hexahedron::from_values(|k| values[k]);
    assert_eq!(
        again
            .values()
            .iter()
            .map(|x| x.to_bits())
            .collect::<Vec<_>>(),
        values.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
    );
}
