use super::DeviceModel;
use crate::{
    constitutive::solid::hyperelastic::{NeoHookean, SaintVenantKirchhoff},
    mechanics::test::get_deformation_gradient,
    units::Stress,
};

fn is_a_kernel_name<M: DeviceModel>() {
    assert!(!M::NAME.is_empty());
    assert!(
        M::NAME
            .chars()
            .all(|character| character.is_ascii_lowercase() || character == '_'),
        "{}",
        M::NAME
    );
}

fn is_made_again_by_its_parameters<M: DeviceModel>(model: &M) {
    let parameters = model.parameters();
    let again = M::from_parameters(&parameters);
    assert_eq!(again.parameters(), parameters);
    let deformation_gradient = get_deformation_gradient();
    let stress = model.cauchy_stress(&deformation_gradient).unwrap();
    let stress_again = again.cauchy_stress(&deformation_gradient).unwrap();
    assert!(stress == stress_again);
}

#[test]
fn neo_hookean_is_made_again_by_its_parameters() {
    let model = NeoHookean {
        shear_modulus: Stress::pascals(3.0),
        bulk_modulus: Stress::pascals(13.0),
    };
    is_a_kernel_name::<NeoHookean>();
    is_made_again_by_its_parameters(&model);
    assert_eq!(model.parameters(), [3.0, 13.0]);
}

#[test]
fn saint_venant_kirchhoff_is_made_again_by_its_parameters() {
    let model = SaintVenantKirchhoff {
        shear_modulus: Stress::pascals(3.0),
        bulk_modulus: Stress::pascals(13.0),
    };
    is_a_kernel_name::<SaintVenantKirchhoff>();
    is_made_again_by_its_parameters(&model);
    assert_eq!(model.parameters(), [3.0, 13.0]);
}

#[test]
fn the_names_of_the_models_differ() {
    assert_ne!(NeoHookean::NAME, SaintVenantKirchhoff::NAME);
}
