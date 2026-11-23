#![cfg(all(feature = "gpu", feature = "fem"))]

mod common;

use common::{
    bits, components, deformed, density, describe, flat, lattice, on_device, slab, steel,
};
use conspire::{
    constitutive::solid::hyperelastic::SaintVenantKirchhoff,
    fem::{
        Blocks, Model, NodalCoordinates, NodalReferenceCoordinates, NodalVelocities,
        block::Block,
        boundary::{BoundaryConditions, ConstrainAt, Displacement, Solid},
        element::{linear::Hexahedron, stabilized::linear::Hexahedron as StabilizedHexahedron},
        solid::{ElasticExplicitDynamics, Execution, Lumped, MassChoice, Unchecked},
    },
    gpu::{DeviceBlocks, GpuError, Resident},
    math::{
        Quantity, Tensor,
        assert::{Assert, AssertionError},
        integrate::VelocityVerlet,
    },
    units::{Density, Time},
};

type Steel = Block<SaintVenantKirchhoff, Hexahedron, 8, 3, 8, 8, Quantity<Density>>;
type StabilizedSteel =
    Block<SaintVenantKirchhoff, StabilizedHexahedron, 1, 3, 8, 8, Quantity<Density>>;

const COLUMNS: usize = 3;
const LAYERS: usize = 4;

const DT: f64 = 1.0 / 1_048_576.0;

struct Work<B> {
    model: Model<B, 3>,
    coordinates: Vec<[f64; 3]>,
    held: Vec<usize>,
}

fn held() -> Vec<usize> {
    (0..3 * (COLUMNS + 1) * (COLUMNS + 1)).collect()
}

fn steelwork() -> Work<Steel> {
    let coordinates = lattice(COLUMNS, LAYERS);
    let reference = NodalReferenceCoordinates::from(coordinates.clone());
    let block = Steel::from((steel(), density(), slab(COLUMNS, 0..LAYERS), &reference));
    Work {
        model: (block, reference).into(),
        coordinates,
        held: held(),
    }
}

fn both_kinds() -> Work<Blocks<Steel, StabilizedSteel>> {
    let coordinates = lattice(COLUMNS, LAYERS);
    let reference = NodalReferenceCoordinates::from(coordinates.clone());
    let blocks = Blocks(
        Steel::from((steel(), density(), slab(COLUMNS, 0..2), &reference)),
        StabilizedSteel::from((steel(), density(), slab(COLUMNS, 2..LAYERS), &reference)),
    );
    Work {
        model: (blocks, reference).into(),
        coordinates,
        held: held(),
    }
}

fn moving(coordinates: &[[f64; 3]]) -> NodalVelocities<3> {
    coordinates
        .iter()
        .map(|&[x, y, z]| {
            [
                0.7 * (2.0 * y + z).sin(),
                0.4 * (3.0 * z).cos(),
                0.9 * x * y,
            ]
        })
        .collect::<Vec<_>>()
        .into()
}

fn close(found: &[f64], expected: &[f64], what: &str) -> Result<(), AssertionError> {
    assert_eq!(found.len(), expected.len());
    let largest = expected
        .iter()
        .fold(0.0_f64, |largest, x| largest.max(x.abs()));
    let worst = found
        .iter()
        .zip(expected)
        .fold(0.0_f64, |worst, (found, expected)| {
            worst.max((found - expected).abs())
        });
    assert!(largest > 0.0, "the {what} are nothing");
    if worst <= 1e-9 * largest {
        Ok(())
    } else {
        Err(format!("the {what} differ by {worst} of up to {largest}").into())
    }
}

fn host_run<B>(
    model: &Model<B, 3>,
    steps: usize,
    start: &NodalCoordinates<3>,
    velocities: &NodalVelocities<3>,
) -> Result<[Vec<f64>; 3], AssertionError>
where
    Model<B, 3>: ElasticExplicitDynamics,
    Lumped: MassChoice<Model<B, 3>>,
{
    let base = 0..(COLUMNS + 1) * (COLUMNS + 1);
    let conditions = BoundaryConditions::<Solid<3>>::new()
        .fix_nodes(Displacement::<0>, base.clone())
        .fix_nodes(Displacement::<1>, base.clone())
        .fix_nodes(Displacement::<2>, base);
    let time = (0..=steps)
        .map(|step| Time::seconds(step as f64 * DT))
        .collect::<Vec<_>>();
    let (_, x, v, a) = model.integrate(
        Execution::Serial,
        Unchecked,
        Lumped,
        &VelocityVerlet::new(Time::seconds(DT)),
        &time,
        (start.clone(), velocities.clone()),
        &(),
        &conditions,
    )?;
    let last = |what: &str| format!("the host's run has {what}");
    Ok([
        flat(x.iter().last().ok_or_else(|| last("coordinates"))?),
        flat(v.iter().last().ok_or_else(|| last("velocities"))?),
        flat(a.iter().last().ok_or_else(|| last("accelerations"))?),
    ])
}

fn state(resident: &Resident) -> Result<[Vec<u64>; 3], AssertionError> {
    let (x, v, a) = resident.dynamics_state().map_err(describe)?;
    Ok([bits(&flat(&x)), bits(&flat(&v)), bits(&flat(&a))])
}

#[test]
fn a_step_on_the_device_is_the_step_of_the_integrator_to_the_bit() -> Result<(), AssertionError> {
    let Work {
        model,
        coordinates,
        held,
    } = steelwork();
    let Some(mut resident) = on_device(&model, false) else {
        return Ok(());
    };
    let masses = Lumped.masses(&model);
    let per_node = masses.iter().map(|mass| mass.value()).collect::<Vec<_>>();
    resident
        .start_dynamics(
            &masses,
            &held,
            &deformed(&coordinates),
            &moving(&coordinates),
        )
        .map_err(describe)?;
    for step in 0..10 {
        let (x, v, a) = resident.dynamics_state().map_err(describe)?;
        let (x, v, a) = (flat(&x), flat(&v), flat(&a));
        resident.steps(DT, 1).map_err(describe)?;
        let (x_next, v_next, a_next) = resident.dynamics_state().map_err(describe)?;
        let forces = components(&resident.download().map_err(describe)?);
        let half = (0..x.len())
            .map(|i| a[i] * (DT * 0.5) + v[i])
            .collect::<Vec<_>>();
        let expected_x = (0..x.len())
            .map(|i| half[i] * DT + x[i])
            .collect::<Vec<_>>();
        let expected_a = (0..x.len())
            .map(|i| {
                if held.contains(&i) {
                    0.0
                } else {
                    -forces[i] / per_node[i / 3]
                }
            })
            .collect::<Vec<_>>();
        let expected_v = (0..x.len())
            .map(|i| expected_a[i] * (DT * 0.5) + half[i])
            .collect::<Vec<_>>();
        assert_eq!(
            bits(&flat(&x_next)),
            bits(&expected_x),
            "coordinates, step {step}"
        );
        assert_eq!(
            bits(&flat(&a_next)),
            bits(&expected_a),
            "accelerations, step {step}"
        );
        assert_eq!(
            bits(&flat(&v_next)),
            bits(&expected_v),
            "velocities, step {step}"
        );
    }
    Ok(())
}

#[test]
fn steps_on_the_device_are_those_of_the_integrator_of_the_host() -> Result<(), AssertionError> {
    let Work {
        model,
        coordinates,
        held,
    } = steelwork();
    let steps = 20;
    let (start, velocities) = (deformed(&coordinates), moving(&coordinates));
    let [x, v, a] = host_run(&model, steps, &start, &velocities)?;
    let Some(mut resident) = on_device(&model, false) else {
        return Ok(());
    };
    resident
        .start_dynamics(&Lumped.masses(&model), &held, &start, &velocities)
        .map_err(describe)?;
    resident.steps(DT, steps).map_err(describe)?;
    let (found_x, found_v, found_a) = resident.dynamics_state().map_err(describe)?;
    close(&flat(&found_x), &x, "coordinates")?;
    close(&flat(&found_v), &v, "velocities")?;
    close(&flat(&found_a), &a, "accelerations")?;
    let (moved, started) = (flat(&found_x), flat(&start));
    for &dof in &held {
        Assert::eq(moved[dof], started[dof])?;
    }
    Ok(())
}

#[test]
fn blocks_of_both_kinds_take_the_steps_of_the_integrator_of_the_host() -> Result<(), AssertionError>
{
    let work = both_kinds();
    let steps = 20;
    let (start, velocities) = (deformed(&work.coordinates), moving(&work.coordinates));
    let [x, v, a] = host_run(&work.model, steps, &start, &velocities)?;
    let Some(mut resident) = on_device(&work.model, false) else {
        return Ok(());
    };
    resident
        .start_dynamics(&Lumped.masses(&work.model), &work.held, &start, &velocities)
        .map_err(describe)?;
    resident.steps(DT, steps).map_err(describe)?;
    let (found_x, found_v, found_a) = resident.dynamics_state().map_err(describe)?;
    close(&flat(&found_x), &x, "coordinates")?;
    close(&flat(&found_v), &v, "velocities")?;
    close(&flat(&found_a), &a, "accelerations")
}

#[test]
fn steps_taken_together_are_those_taken_one_at_a_time() -> Result<(), AssertionError> {
    let Work {
        model,
        coordinates,
        held,
    } = steelwork();
    let Some(mut resident) = on_device(&model, true) else {
        return Ok(());
    };
    let masses = Lumped.masses(&model);
    let (start, velocities) = (deformed(&coordinates), moving(&coordinates));
    resident
        .start_dynamics(&masses, &held, &start, &velocities)
        .map_err(describe)?;
    resident.steps(DT, 6).map_err(describe)?;
    let together = state(&resident)?;
    resident
        .start_dynamics(&masses, &held, &start, &velocities)
        .map_err(describe)?;
    for _ in 0..3 {
        resident.steps(DT, 2).map_err(describe)?;
    }
    assert_eq!(state(&resident)?, together);
    resident
        .start_dynamics(&masses, &held, &start, &velocities)
        .map_err(describe)?;
    for _ in 0..6 {
        resident.steps(DT, 1).map_err(describe)?;
    }
    assert_eq!(state(&resident)?, together);
    Ok(())
}

#[test]
fn an_element_with_an_error_in_a_run_of_dynamics_is_an_error() -> Result<(), AssertionError> {
    let Work {
        model,
        coordinates,
        held,
    } = steelwork();
    let Some(mut resident) = on_device(&model, false) else {
        return Ok(());
    };
    let masses = Lumped.masses(&model);
    let velocities = moving(&coordinates);
    let good = deformed(&coordinates);
    let mut bad = good.clone();
    bad[1] = [0.9, 0.9, 0.9].into();
    let error = resident
        .start_dynamics(&masses, &held, &bad, &velocities)
        .expect_err("the start has an error");
    assert!(matches!(error, GpuError::Element { .. }), "{error}");
    resident
        .start_dynamics(&masses, &held, &good, &velocities)
        .map_err(describe)?;
    resident.steps(DT, 2).map_err(describe)?;
    resident.set_coordinates(&bad).map_err(describe)?;
    let error = resident.steps(DT, 1).expect_err("the step has an error");
    assert!(matches!(error, GpuError::Element { .. }), "{error}");
    Ok(())
}

#[test]
fn what_is_not_a_run_of_the_model_is_an_error() -> Result<(), AssertionError> {
    let Work {
        model,
        coordinates,
        held,
    } = steelwork();
    let Some(mut resident) = on_device(&model, false) else {
        return Ok(());
    };
    let error = resident
        .steps(DT, 1)
        .expect_err("a run that is not started is stepped");
    assert!(matches!(error, GpuError::Invalid(_)), "{error}");
    let error = resident
        .dynamics_state()
        .expect_err("the state of a run that is not started");
    assert!(matches!(error, GpuError::Invalid(_)), "{error}");
    let mut outside = held.clone();
    outside.push(3 * coordinates.len());
    let error = resident
        .start_dynamics(
            &Lumped.masses(&model),
            &outside,
            &deformed(&coordinates),
            &moving(&coordinates),
        )
        .expect_err("a degree of freedom that the model does not have is held");
    assert!(matches!(error, GpuError::Invalid(_)), "{error}");
    Ok(())
}

#[test]
fn the_blocks_of_the_work_are_those_that_a_device_takes() {
    fn takes<B: DeviceBlocks>(_: &Model<B, 3>) {}
    takes(&steelwork().model);
    takes(&both_kinds().model);
}
