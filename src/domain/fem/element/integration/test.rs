use super::IntegrationData;
use crate::{
    fem::element::{ElementNodalReferenceCoordinates, FiniteElement, linear::LinearElement},
    math::{
        Quantity,
        assert::{Assert, AssertionError},
    },
    units::Volume,
};

const HEXAHEDRON: [[f64; 3]; 8] = [
    [0.0, 0.0, 0.0],
    [2.0, 0.0, 0.0],
    [2.3, 2.1, 0.0],
    [0.0, 2.0, 0.0],
    [0.0, 0.0, 2.0],
    [2.0, 0.2, 2.2],
    [3.0, 2.5, 2.4],
    [0.1, 2.0, 2.0],
];

#[test]
fn the_data_of_an_element_are_those_that_it_stores() -> Result<(), AssertionError> {
    let element = LinearElement::from(ElementNodalReferenceCoordinates::from(HEXAHEDRON));
    (0..8).try_for_each(|point| {
        Assert::default().eq_within_tols(
            element.gradient_vectors_at(point),
            &element.gradient_vectors()[point],
        )?;
        Assert::default().eq_within_tols(
            element.integration_weight_at(point),
            &element.integration_weights()[point],
        )
    })
}

#[test]
fn the_weights_of_the_points_of_an_element_add_up_to_its_volume() -> Result<(), AssertionError> {
    let element = LinearElement::from(ElementNodalReferenceCoordinates::from(HEXAHEDRON));
    let sum: Quantity<Volume> = (0..8)
        .map(|point| element.integration_weight_at(point))
        .sum();
    Assert::default().eq_within_tols(sum, &element.volume())
}
