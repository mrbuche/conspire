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
            Quantity, Tensor,
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
            .nodal_forces(&model(), &coordinates().into())?;
        forces.iter().try_for_each(|force| {
            (0..3).try_for_each(|i| Assert::default().eq_within_tols(force[i].value(), 0.0))
        })
    }
    #[test]
    fn forces_balance_for_any_state() -> Result<(), AssertionError> {
        let forces =
            Hexahedron::from(reference_coordinates()).nodal_forces(&model(), &inhomogeneous())?;
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
            Hexahedron::from(reference_coordinates()).nodal_forces(&model(), &homogeneous())?,
            &FullHexahedron::from(reference_coordinates())
                .nodal_forces(&model(), &homogeneous())?,
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
                .nodal_stiffnesses(&model(), &homogeneous())?,
            &linear_displacements(),
        );
        let full = stiffnesses_times(
            &FullHexahedron::from(reference_coordinates())
                .nodal_stiffnesses(&model(), &homogeneous())?,
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
    fn hourglass_stiffness_is_the_fraction_of_the_cubes_in_a_volume_preserving_mode()
    -> Result<(), AssertionError> {
        use crate::fem::block::element::{
            ElementNodalReferenceCoordinates, stabilized::HOURGLASS_PARAMETER,
        };
        let cube: [[f64; 3]; N] = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
        ];
        let mode = [1.0, 1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0];
        let displacements: [[f64; 3]; N] = from_fn(|a| [mode[a], 0.0, 0.0]);
        let energy = |stiffnesses: &ElementNodalStiffnessesSolid<N>| {
            let forces = stiffnesses_times(stiffnesses, &displacements);
            (0..N)
                .map(|a| {
                    (0..3)
                        .map(|i| displacements[a][i] * forces[a][i])
                        .sum::<f64>()
                })
                .sum::<f64>()
        };
        let reference: ElementNodalReferenceCoordinates<N> = cube.into();
        let current: ElementNodalCoordinates<N> = cube.into();
        let full = FullHexahedron::from(reference.clone()).nodal_stiffnesses(&model(), &current)?;
        let hourglass = Hexahedron::from(reference).hourglass_stiffnesses(model().shear_modulus());
        Assert::default().eq_within_tols(energy(&hourglass) / HOURGLASS_PARAMETER, energy(&full))
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
                    .helmholtz_free_energy(&model(), &homogeneous())?
                    .value(),
                FullHexahedron::from(reference_coordinates())
                    .helmholtz_free_energy(&model(), &homogeneous())?
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
    mod mass {
        use super::*;
        use crate::{
            fem::block::element::{
                FiniteElement, mass::LumpedMassFiniteElement, time_scale::fastest_time_scale,
            },
            units::Density,
        };
        const DENSITY: Quantity<Density> = Density::kilograms_per_cubic_meter(7.8e3);
        #[test]
        fn lumped_masses_are_those_of_the_fully_integrated_element() -> Result<(), AssertionError> {
            let densities = [DENSITY].into();
            let full_densities = [DENSITY; 8].into();
            Assert::default().eq_within_tols(
                Hexahedron::from(reference_coordinates()).nodal_lumped_masses(&densities),
                FullHexahedron::from(reference_coordinates()).nodal_lumped_masses(&full_densities),
            )
        }
        #[test]
        fn lumped_masses_add_up_to_the_mass() -> Result<(), AssertionError> {
            let element = Hexahedron::from(reference_coordinates());
            let masses = element.nodal_lumped_masses(&[DENSITY].into());
            Assert::default().eq_within_tols(
                masses.iter().map(|mass| mass.value()).sum::<f64>(),
                DENSITY.value() * element.volume().value(),
            )
        }
        #[test]
        fn fastest_time_scale_is_finite() {
            let element = Hexahedron::from(reference_coordinates());
            let time_scale = fastest_time_scale(
                &element.nodal_stiffnesses(&model(), &homogeneous()).unwrap(),
                &element.nodal_lumped_masses(&[DENSITY].into()),
            )
            .value();
            assert!(time_scale.is_finite() && time_scale > 0.0)
        }
    }
}

mod block {
    use super::{FullHexahedron, Hexahedron};
    use crate::{
        constitutive::solid::{
            elastic::test::{BULK_MODULUS, SHEAR_MODULUS},
            hyperelastic::NeoHookean,
        },
        fem::{
            NodalCoordinates, NodalReferenceCoordinates, block::Block,
            solid::elastic::ElasticElements,
        },
        math::assert::{Assert, AssertionError},
    };
    use core::array::from_fn;

    type Stabilized = Block<NeoHookean, Hexahedron, 1, 3, 8, 8>;
    type Full = Block<NeoHookean, FullHexahedron, 8, 3, 8, 8>;
    const NODES: usize = 27;
    const INTERIOR_NODE: usize = 13;
    fn model() -> NeoHookean {
        NeoHookean {
            bulk_modulus: BULK_MODULUS,
            shear_modulus: SHEAR_MODULUS,
        }
    }
    fn connectivity() -> Vec<[usize; 8]> {
        (0..8)
            .map(|element| {
                let (i, j, k) = (element % 2, (element / 2) % 2, element / 4);
                let node = |di: usize, dj: usize, dk: usize| (i + di) + 3 * (j + dj) + 9 * (k + dk);
                [
                    node(0, 0, 0),
                    node(1, 0, 0),
                    node(1, 1, 0),
                    node(0, 1, 0),
                    node(0, 0, 1),
                    node(1, 0, 1),
                    node(1, 1, 1),
                    node(0, 1, 1),
                ]
            })
            .collect()
    }
    fn reference_array() -> [[f64; 3]; NODES] {
        from_fn(|node| {
            let grid = [node % 3, (node / 3) % 3, node / 9];
            from_fn(|i| 0.5 * grid[i] as f64 - 0.5 + 0.04 * ((7 * node + 3 * i) as f64).sin())
        })
    }
    fn reference_coordinates() -> NodalReferenceCoordinates<3> {
        reference_array().into()
    }
    fn deformed(deformation_gradient: [[f64; 3]; 3], translation: [f64; 3]) -> NodalCoordinates<3> {
        let reference = reference_array();
        let deformed: [[f64; 3]; NODES] = from_fn(|node| {
            from_fn(|i| {
                (0..3)
                    .map(|j| deformation_gradient[i][j] * reference[node][j])
                    .sum::<f64>()
                    + translation[i]
            })
        });
        deformed.into()
    }
    fn homogeneous() -> NodalCoordinates<3> {
        deformed(
            [[1.1, 0.1, 0.0], [0.0, 0.9, 0.2], [0.1, 0.0, 1.05]],
            [0.3, -0.2, 0.1],
        )
    }
    #[test]
    fn forces_match_full_integration_for_a_homogeneous_deformation() -> Result<(), AssertionError> {
        let stabilized = Stabilized::from((model(), connectivity(), &reference_coordinates()));
        let full = Full::from((model(), connectivity(), &reference_coordinates()));
        Assert::default().eq_within_tols(
            stabilized.nodal_forces(&homogeneous())?,
            full.nodal_forces(&homogeneous())?,
        )
    }
    #[test]
    fn interior_node_is_in_equilibrium_for_a_homogeneous_deformation() -> Result<(), AssertionError>
    {
        let stabilized = Stabilized::from((model(), connectivity(), &reference_coordinates()));
        let forces = stabilized.nodal_forces(&homogeneous())?;
        (0..3).try_for_each(|i| {
            Assert::default().eq_within_tols(forces[INTERIOR_NODE][i].value(), 0.0)
        })
    }
    #[test]
    fn forces_vanish_for_a_rigid_motion() -> Result<(), AssertionError> {
        let stabilized = Stabilized::from((model(), connectivity(), &reference_coordinates()));
        let (sin, cos) = 0.3_f64.sin_cos();
        let forces = stabilized.nodal_forces(&deformed(
            [[cos, -sin, 0.0], [sin, cos, 0.0], [0.0, 0.0, 1.0]],
            [0.5, -0.4, 0.2],
        ))?;
        (0..NODES).try_for_each(|node| {
            (0..3).try_for_each(|i| Assert::default().eq_within_tols(forces[node][i].value(), 0.0))
        })
    }
}
