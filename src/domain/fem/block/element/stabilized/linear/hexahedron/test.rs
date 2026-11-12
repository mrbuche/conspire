use crate::{
    fem::block::element::{
        ElementNodalReferenceCoordinates, FiniteElement,
        linear::Hexahedron as FullHexahedron,
        stabilized::linear::hexahedron::{Hexahedron, N},
    },
    math::{
        Tensor,
        assert::{Assert, AssertionError},
    },
};
use core::array::from_fn;

fn reference_coordinates() -> ElementNodalReferenceCoordinates<N> {
    [
        [0.0, 0.1, -0.1],
        [1.1, 0.0, 0.1],
        [1.2, 0.9, 0.0],
        [-0.1, 1.0, 0.2],
        [0.1, -0.1, 1.0],
        [1.0, 0.1, 1.2],
        [0.9, 1.1, 0.9],
        [0.0, 0.8, 1.1],
    ]
    .into()
}

fn coordinates() -> [[f64; 3]; N] {
    let reference_coordinates = reference_coordinates();
    from_fn(|a| [0, 1, 2].map(|i| reference_coordinates[a][i].value()))
}

fn mean_gradient() -> [[f64; 3]; N] {
    let element = Hexahedron::from(reference_coordinates());
    from_fn(|a| {
        let gradient_vector = &element.gradient_vectors()[0][a];
        [0, 1, 2].map(|i| gradient_vector[i].value())
    })
}

fn hourglass_matrix() -> [[f64; N]; N] {
    let element = Hexahedron::from(reference_coordinates());
    from_fn(|a| from_fn(|b| element.hourglass_matrix()[a][b].value()))
}

#[test]
fn volume_is_that_of_the_fully_integrated_element() -> Result<(), AssertionError> {
    Assert::default().eq_within_tols(
        Hexahedron::from(reference_coordinates()).volume().value(),
        FullHexahedron::from(reference_coordinates())
            .volume()
            .value(),
    )
}

#[test]
fn nodal_volumes_add_up_to_the_volume() -> Result<(), AssertionError> {
    let element = Hexahedron::from(reference_coordinates());
    Assert::default().eq_within_tols(
        element
            .nodal_volumes()
            .iter()
            .map(|volume| volume.value())
            .sum::<f64>(),
        element.volume().value(),
    )
}

#[test]
fn mean_gradient_sums_to_zero() -> Result<(), AssertionError> {
    let gradient = mean_gradient();
    (0..3).try_for_each(|i| {
        Assert::default().eq_within_tols(gradient.iter().map(|g| g[i]).sum::<f64>(), 0.0)
    })
}

#[test]
fn mean_gradient_reproduces_the_identity() -> Result<(), AssertionError> {
    let gradient = mean_gradient();
    let x = coordinates();
    (0..3).try_for_each(|i| {
        (0..3).try_for_each(|j| {
            Assert::default().eq_within_tols(
                (0..N).map(|a| x[a][i] * gradient[a][j]).sum::<f64>(),
                if i == j { 1.0 } else { 0.0 },
            )
        })
    })
}

#[test]
fn hourglass_matrix_is_symmetric() -> Result<(), AssertionError> {
    let h = hourglass_matrix();
    (0..N).try_for_each(|a| {
        (0..N).try_for_each(|b| Assert::default().eq_within_tols(h[a][b], h[b][a]))
    })
}

#[test]
fn hourglass_matrix_ignores_constant_and_linear_fields() -> Result<(), AssertionError> {
    let h = hourglass_matrix();
    let x = coordinates();
    (0..N).try_for_each(|a| {
        Assert::default().eq_within_tols(h[a].iter().sum::<f64>(), 0.0)?;
        (0..3).try_for_each(|i| {
            Assert::default().eq_within_tols((0..N).map(|b| h[a][b] * x[b][i]).sum::<f64>(), 0.0)
        })
    })
}

#[test]
fn hourglass_matrix_is_positive_on_an_hourglass_mode() {
    let h = hourglass_matrix();
    let mode = [1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0];
    let energy = (0..N)
        .map(|a| (0..N).map(|b| mode[a] * h[a][b] * mode[b]).sum::<f64>())
        .sum::<f64>();
    assert!(energy > 0.0)
}

#[test]
fn debug_names_the_element() {
    assert_eq!(
        format!("{:?}", Hexahedron::from(reference_coordinates())),
        "StabilizedHexahedron { integration points: 1, nodes: 8 }"
    )
}

mod elastic {
    use super::{FullHexahedron, Hexahedron, N, coordinates, reference_coordinates};
    use crate::{
        constitutive::solid::{
            Solid,
            elastic::test::{BULK_MODULUS, SHEAR_MODULUS},
            hyperelastic::NeoHookean,
        },
        domain::block::element::solid::elastic::ElasticElement,
        fem::block::element::{
            ElementNodalCoordinates,
            solid::{ElementNodalForcesSolid, ElementNodalStiffnessesSolid},
        },
        math::{
            Tensor,
            assert::{Assert, AssertionError},
        },
    };
    use core::array::from_fn;
    fn model() -> NeoHookean {
        NeoHookean {
            bulk_modulus: BULK_MODULUS,
            shear_modulus: SHEAR_MODULUS,
        }
    }
    fn homogeneous_array() -> [[f64; 3]; N] {
        let deformation_gradient = [[1.1, 0.1, 0.0], [0.0, 0.9, 0.2], [0.1, 0.0, 1.05]];
        let translation = [0.3, -0.2, 0.1];
        let reference = coordinates();
        from_fn(|a| {
            from_fn(|i| {
                (0..3)
                    .map(|j| deformation_gradient[i][j] * reference[a][j])
                    .sum::<f64>()
                    + translation[i]
            })
        })
    }
    fn homogeneous() -> ElementNodalCoordinates<N> {
        homogeneous_array().into()
    }
    fn inhomogeneous_array() -> [[f64; 3]; N] {
        let mut deformed = homogeneous_array();
        deformed[6][0] += 0.07;
        deformed[2][1] -= 0.05;
        deformed[5][2] += 0.06;
        deformed
    }
    fn inhomogeneous() -> ElementNodalCoordinates<N> {
        inhomogeneous_array().into()
    }
    #[test]
    fn forces_are_zero_in_the_reference_configuration() -> Result<(), AssertionError> {
        let forces = Hexahedron::from(reference_coordinates())
            .nodal_forces(&model(), &coordinates().into())
            .unwrap();
        forces.iter().try_for_each(|force| {
            (0..3).try_for_each(|i| Assert::default().eq_within_tols(force[i].value(), 0.0))
        })
    }
    #[test]
    fn forces_balance_for_any_state() -> Result<(), AssertionError> {
        let forces = Hexahedron::from(reference_coordinates())
            .nodal_forces(&model(), &inhomogeneous())
            .unwrap();
        (0..3).try_for_each(|i| {
            Assert::default().eq_within_tols(
                forces.iter().map(|force| force[i].value()).sum::<f64>(),
                0.0,
            )
        })
    }
    #[test]
    fn forces_match_full_integration_for_a_homogeneous_deformation() -> Result<(), AssertionError> {
        Assert::default().eq_within_tols(
            Hexahedron::from(reference_coordinates())
                .nodal_forces(&model(), &homogeneous())
                .unwrap(),
            &FullHexahedron::from(reference_coordinates())
                .nodal_forces(&model(), &homogeneous())
                .unwrap(),
        )
    }
    fn forces_array(forces: &ElementNodalForcesSolid<N>) -> [[f64; 3]; N] {
        from_fn(|a| from_fn(|i| forces[a][i].value()))
    }
    fn stiffnesses_times(
        stiffnesses: &ElementNodalStiffnessesSolid<N>,
        displacements: &[[f64; 3]; N],
    ) -> [[f64; 3]; N] {
        from_fn(|a| {
            from_fn(|i| {
                (0..N)
                    .map(|b| {
                        (0..3)
                            .map(|j| stiffnesses[a][b][i][j].value() * displacements[b][j])
                            .sum::<f64>()
                    })
                    .sum()
            })
        })
    }
    fn linear_displacements() -> [[f64; 3]; N] {
        let gradient = [[0.02, -0.01, 0.0], [0.01, 0.03, 0.02], [0.0, -0.02, 0.01]];
        let reference = coordinates();
        from_fn(|a| from_fn(|i| (0..3).map(|j| gradient[i][j] * reference[a][j]).sum()))
    }
    #[test]
    fn stiffnesses_match_full_integration_on_a_linear_displacement() -> Result<(), AssertionError> {
        let stabilized = stiffnesses_times(
            &Hexahedron::from(reference_coordinates())
                .nodal_stiffnesses(&model(), &homogeneous())
                .unwrap(),
            &linear_displacements(),
        );
        let full = stiffnesses_times(
            &FullHexahedron::from(reference_coordinates())
                .nodal_stiffnesses(&model(), &homogeneous())
                .unwrap(),
            &linear_displacements(),
        );
        (0..N).try_for_each(|a| {
            (0..3).try_for_each(|i| Assert::default().eq_within_tols(stabilized[a][i], full[a][i]))
        })
    }
    #[test]
    fn hourglass_forces_vanish_for_a_homogeneous_deformation() -> Result<(), AssertionError> {
        let forces = forces_array(
            &Hexahedron::from(reference_coordinates())
                .hourglass_forces(model().shear_modulus(), &homogeneous()),
        );
        forces
            .iter()
            .flatten()
            .try_for_each(|force| Assert::default().eq_within_tols(*force, 0.0))
    }
    #[test]
    fn hourglass_forces_resist_an_hourglass_mode() {
        let mode = [1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0];
        let mut deformed = coordinates();
        deformed
            .iter_mut()
            .zip(mode)
            .for_each(|(coordinate, mode_a)| coordinate[0] += 0.05 * mode_a);
        let forces = forces_array(
            &Hexahedron::from(reference_coordinates())
                .hourglass_forces(model().shear_modulus(), &deformed.into()),
        );
        let work = (0..N).map(|a| forces[a][0] * 0.05 * mode[a]).sum::<f64>();
        assert!(work > 0.0)
    }
    #[test]
    fn hourglass_stiffnesses_are_the_derivative_of_the_hourglass_forces()
    -> Result<(), AssertionError> {
        let element = Hexahedron::from(reference_coordinates());
        let shear_modulus = model().shear_modulus();
        let change: [[f64; 3]; N] = from_fn(|a| from_fn(|i| 0.01 * ((3 * a + i) as f64).sin()));
        let state = inhomogeneous_array();
        let perturbed: [[f64; 3]; N] = from_fn(|a| from_fn(|i| state[a][i] + change[a][i]));
        let before = forces_array(&element.hourglass_forces(shear_modulus, &state.into()));
        let after = forces_array(&element.hourglass_forces(shear_modulus, &perturbed.into()));
        let predicted = stiffnesses_times(&element.hourglass_stiffnesses(shear_modulus), &change);
        (0..N).try_for_each(|a| {
            (0..3).try_for_each(|i| {
                Assert::default().eq_within_tols(after[a][i] - before[a][i], predicted[a][i])
            })
        })
    }
    #[test]
    fn hourglass_forces_rotate_with_the_element() -> Result<(), AssertionError> {
        let element = Hexahedron::from(reference_coordinates());
        let shear_modulus = model().shear_modulus();
        let (sin, cos) = 0.3_f64.sin_cos();
        let rotation = [[cos, -sin, 0.0], [sin, cos, 0.0], [0.0, 0.0, 1.0]];
        let state = inhomogeneous_array();
        let rotated: [[f64; 3]; N] =
            from_fn(|a| from_fn(|i| (0..3).map(|j| rotation[i][j] * state[a][j]).sum()));
        let forces = forces_array(&element.hourglass_forces(shear_modulus, &state.into()));
        let rotated_forces =
            forces_array(&element.hourglass_forces(shear_modulus, &rotated.into()));
        (0..N).try_for_each(|a| {
            (0..3).try_for_each(|i| {
                Assert::default().eq_within_tols(
                    rotated_forces[a][i],
                    (0..3).map(|j| rotation[i][j] * forces[a][j]).sum::<f64>(),
                )
            })
        })
    }
    mod hyperelastic {
        use super::*;
        use crate::domain::block::element::solid::hyperelastic::HyperelasticElement;
        #[test]
        fn energy_matches_full_integration_for_a_homogeneous_deformation()
        -> Result<(), AssertionError> {
            Assert::default().eq_within_tols(
                Hexahedron::from(reference_coordinates())
                    .helmholtz_free_energy(&model(), &homogeneous())
                    .unwrap()
                    .value(),
                FullHexahedron::from(reference_coordinates())
                    .helmholtz_free_energy(&model(), &homogeneous())
                    .unwrap()
                    .value(),
            )
        }
        #[test]
        fn hourglass_energy_vanishes_for_a_homogeneous_deformation() -> Result<(), AssertionError> {
            Assert::default().eq_within_tols(
                Hexahedron::from(reference_coordinates())
                    .hourglass_helmholtz_free_energy(model().shear_modulus(), &homogeneous())
                    .value(),
                0.0,
            )
        }
        #[test]
        fn hourglass_energy_rises_on_an_hourglass_mode() {
            let mode = [1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0];
            let mut deformed = coordinates();
            deformed
                .iter_mut()
                .zip(mode)
                .for_each(|(coordinate, mode_a)| coordinate[0] += 0.05 * mode_a);
            let energy = Hexahedron::from(reference_coordinates())
                .hourglass_helmholtz_free_energy(model().shear_modulus(), &deformed.into())
                .value();
            assert!(energy > 0.0)
        }
        #[test]
        fn hourglass_forces_are_the_gradient_of_the_hourglass_energy() -> Result<(), AssertionError>
        {
            let element = Hexahedron::from(reference_coordinates());
            let shear_modulus = model().shear_modulus();
            let change: [[f64; 3]; N] = from_fn(|a| from_fn(|i| 0.01 * ((3 * a + i) as f64).sin()));
            let state = inhomogeneous_array();
            let plus: [[f64; 3]; N] = from_fn(|a| from_fn(|i| state[a][i] + change[a][i]));
            let minus: [[f64; 3]; N] = from_fn(|a| from_fn(|i| state[a][i] - change[a][i]));
            let forces = forces_array(&element.hourglass_forces(shear_modulus, &state.into()));
            Assert::default().eq_within_tols(
                element
                    .hourglass_helmholtz_free_energy(shear_modulus, &plus.into())
                    .value()
                    - element
                        .hourglass_helmholtz_free_energy(shear_modulus, &minus.into())
                        .value(),
                2.0 * (0..N)
                    .map(|a| (0..3).map(|i| forces[a][i] * change[a][i]).sum::<f64>())
                    .sum::<f64>(),
            )
        }
    }
}
