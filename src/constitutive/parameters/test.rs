use super::FromParameters;
use crate::{
    constitutive::solid::{
        elastic::Elastic,
        hyperelastic::{NeoHookean, SaintVenantKirchhoff},
    },
    math::assert::{Assert, AssertionError},
    mechanics::test::get_deformation_gradient,
    units::Stress,
};

fn are_equal(found: &[f64], expected: &[f64]) -> Result<(), AssertionError> {
    assert_eq!(found.len(), expected.len());
    found
        .iter()
        .zip(expected)
        .try_for_each(|(found, expected)| Assert::default().eq_within_tols(found, expected))
}

fn is_made_again_by_its_parameters<M: Elastic + FromParameters>(
    model: &M,
    expected: &[f64],
) -> Result<(), AssertionError> {
    let parameters = model.parameters();
    are_equal(&parameters, expected)?;
    let again = M::from_parameters(&parameters);
    are_equal(&again.parameters(), &parameters)?;
    let deformation_gradient = get_deformation_gradient();
    Assert::default().eq_within_tols(
        &model.cauchy_stress(&deformation_gradient)?,
        &again.cauchy_stress(&deformation_gradient)?,
    )
}

fn is_a_kernel_name<M: FromParameters>() {
    assert!(!M::NAME.is_empty());
    assert!(
        M::NAME
            .chars()
            .all(|character| character.is_ascii_lowercase() || character == '_'),
        "{}",
        M::NAME
    );
}

#[test]
fn the_names_of_the_models_are_kernel_names_and_differ() {
    is_a_kernel_name::<NeoHookean>();
    is_a_kernel_name::<SaintVenantKirchhoff>();
    assert_ne!(NeoHookean::NAME, SaintVenantKirchhoff::NAME);
}

#[test]
fn neo_hookean_is_made_again_by_its_parameters() -> Result<(), AssertionError> {
    is_made_again_by_its_parameters(
        &NeoHookean {
            shear_modulus: Stress::pascals(3.0),
            bulk_modulus: Stress::pascals(13.0),
        },
        &[3.0, 13.0],
    )
}

#[test]
fn saint_venant_kirchhoff_is_made_again_by_its_parameters() -> Result<(), AssertionError> {
    is_made_again_by_its_parameters(
        &SaintVenantKirchhoff {
            shear_modulus: Stress::pascals(3.0),
            bulk_modulus: Stress::pascals(13.0),
        },
        &[3.0, 13.0],
    )
}
