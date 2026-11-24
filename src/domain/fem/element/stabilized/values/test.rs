use super::StabilizedElement;
use crate::{
    constitutive::solid::hyperelastic::{NeoHookean, SaintVenantKirchhoff},
    fem::element::{
        ElementNodalCoordinates, ElementNodalReferenceCoordinates, FiniteElement,
        solid::{ElementNodalForcesSolid, elastic::ElasticElement},
    },
    math::{
        Tensor,
        assert::{Assert, AssertionError},
    },
    units::Stress,
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

fn alternating(element: &Hexahedron) -> ElementNodalCoordinates<8> {
    let reference = <Hexahedron as FiniteElement<1, 3, 8, 8>>::parametric_reference();
    let nodes: [[f64; 3]; 8] = from_fn(|a| {
        let sign = if a % 2 == 0 { 1.0 } else { -1.0 };
        let xi = &reference[a];
        [
            1.1 * xi[0].value() + 0.02 * sign,
            0.9 * xi[1].value() - 0.01 * sign,
            1.2 * xi[2].value() + 0.015 * sign,
        ]
    });
    assert!(element.hourglass_forces(Stress::pascals(1.0), &nodes.into())[0][0].value() != 0.0);
    nodes.into()
}

#[test]
fn the_forces_from_the_values_are_the_forces_of_the_element_to_the_bit()
-> Result<(), AssertionError> {
    let element = bent();
    let values = element.values();
    let coordinates = alternating(&element);
    let rubber = NeoHookean {
        shear_modulus: Stress::pascals(3.0e9),
        bulk_modulus: Stress::pascals(13.0e9),
    };
    let steel = SaintVenantKirchhoff {
        shear_modulus: Stress::pascals(80.0e9),
        bulk_modulus: Stress::pascals(160.0e9),
    };
    let bits = |forces: ElementNodalForcesSolid<8>| {
        forces
            .iter()
            .flat_map(|force| force.iter().map(|component| component.value().to_bits()))
            .collect::<Vec<_>>()
    };
    assert_eq!(
        bits(Hexahedron::nodal_forces_from_values(
            |k| values[k],
            &rubber,
            &coordinates
        )?),
        bits(element.nodal_forces(&rubber, &coordinates)?)
    );
    assert_eq!(
        bits(Hexahedron::nodal_forces_from_values(
            |k| values[k],
            &steel,
            &coordinates
        )?),
        bits(element.nodal_forces(&steel, &coordinates)?)
    );
    Ok(())
}

#[test]
fn an_element_has_as_many_values_as_it_says() {
    assert_eq!(bent().values().len(), Hexahedron::VALUES);
    assert_eq!(Hexahedron::VALUES, 24 + 1 + 64);
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
