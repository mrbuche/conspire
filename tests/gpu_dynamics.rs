#![cfg(all(feature = "gpu", feature = "fem"))]

mod common;

use common::{
    FIRST, LAST, RATE, Work, bits, close, components, deformed, density, describe,
    device_forces_of, flat, imposed_by, lattice, loads_on, on_device, schedule_of, slab,
    static_forces, steel, targets_at, time_at, volume_loads_on,
};
use conspire::{
    constitutive::solid::hyperelastic::SaintVenantKirchhoff,
    fem::{
        Blocks, Model, NodalCoordinates, NodalReferenceCoordinates,
        block::Block,
        boundary::{
            BoundaryConditions, ConstrainAt, DeviceForce, DeviceLoads, Displacement, LoadAt, Solid,
        },
        element::{linear::Hexahedron, stabilized::linear::Hexahedron as StabilizedHexahedron},
        solid::{
            ElasticExplicitDynamics, Execution, Lumped, MassChoice, NodalForcesSolid, Unchecked,
            elastic::ElasticElements,
        },
    },
    gpu::{DeviceBlocks, GpuError, Imposed, Resident},
    math::{
        Quantity, QuantityVector, Tensor,
        assert::{Assert, AssertionError},
        integrate::VelocityVerlet,
    },
    units::{Density, Length, Mass, Time},
};

type Steel = Block<SaintVenantKirchhoff, Hexahedron, 8, 3, 8, 8, Quantity<Density>>;
type StabilizedSteel =
    Block<SaintVenantKirchhoff, StabilizedHexahedron, 1, 3, 8, 8, Quantity<Density>>;

const COLUMNS: usize = 3;
const LAYERS: usize = 4;

const DT: f64 = 1.0 / 1_048_576.0;

fn steelwork_of(columns: usize, layers: usize) -> Work<Steel> {
    let coordinates = lattice(columns, layers);
    let reference = NodalReferenceCoordinates::from(coordinates.clone());
    let block = Steel::from((steel(), density(), slab(columns, 0..layers), &reference));
    Work::new((block, reference).into(), coordinates, columns)
}

fn steelwork() -> Work<Steel> {
    steelwork_of(COLUMNS, LAYERS)
}

fn both_kinds() -> Work<Blocks<Steel, StabilizedSteel>> {
    let coordinates = lattice(COLUMNS, LAYERS);
    let reference = NodalReferenceCoordinates::from(coordinates.clone());
    let blocks = Blocks(
        Steel::from((steel(), density(), slab(COLUMNS, 0..2), &reference)),
        StabilizedSteel::from((steel(), density(), slab(COLUMNS, 2..LAYERS), &reference)),
    );
    Work::new((blocks, reference).into(), coordinates, COLUMNS)
}

fn state(resident: &Resident) -> Result<[Vec<u64>; 3], AssertionError> {
    let (x, v, a) = resident.dynamics_state().map_err(describe)?;
    Ok([bits(&flat(&x)), bits(&flat(&v)), bits(&flat(&a))])
}

fn against_the_host<B>(
    work: &Work<B>,
    loads: &(impl LoadAt<NodalCoordinates<3>, NodalForcesSolid<3>> + DeviceLoads),
    moved: bool,
) -> Result<(), AssertionError>
where
    B: DeviceBlocks,
    Model<B, 3>: ElasticExplicitDynamics,
    Lumped: MassChoice<Model<B, 3>, Mass = QuantityVector<Mass>>,
{
    let steps = 20;
    let (start, velocities) = (deformed(&work.coordinates), work.velocities());
    let base = 0..work.layer;
    let top = if moved { work.top.clone() } else { Vec::new() };
    let conditions = BoundaryConditions::<Solid<3>>::new()
        .fix_nodes(Displacement::<0>, base.clone())
        .fix_nodes(Displacement::<1>, base.clone())
        .fix_nodes(Displacement::<2>, base)
        .prescribe_nodes_with(Displacement::<0>, top, |time| {
            Length::meters(RATE * time.value())
        });
    let time = (0..=steps).map(|row| time_at(row, DT)).collect::<Vec<_>>();
    let (_, x, v, a) = work.model.integrate(
        Execution::Serial,
        Unchecked,
        Lumped,
        &VelocityVerlet::new(Time::seconds(DT)),
        &time,
        (start.clone(), velocities.clone()),
        loads,
        &conditions,
    )?;
    let imposed = imposed_by(work, moved);
    let Some(mut resident) = on_device(&work.model, false) else {
        return Ok(());
    };
    let schedule = schedule_of(work, &start, DT);
    resident
        .start_dynamics(
            &Lumped.masses(&work.model),
            &start,
            &velocities,
            &imposed,
            loads,
            &schedule,
        )
        .map_err(describe)?;
    resident
        .steps(DT, steps, loads, &schedule)
        .map_err(describe)?;
    let (found_x, found_v, found_a) = resident.dynamics_state().map_err(describe)?;
    let last = |what: &str| format!("the host's run has {what}");
    let (last_x, last_v, last_a) = (
        flat(x.iter().last().ok_or_else(|| last("coordinates"))?),
        flat(v.iter().last().ok_or_else(|| last("velocities"))?),
        flat(a.iter().last().ok_or_else(|| last("accelerations"))?),
    );
    close(&flat(&found_x), &last_x, "coordinates")?;
    close(&flat(&found_v), &last_v, "velocities")?;
    close(&flat(&found_a), &last_a, "accelerations")?;
    let (found_x, started) = (flat(&found_x), flat(&start));
    for &dof in &work.fixed {
        Assert::eq(found_x[dof], started[dof])?;
    }
    if moved {
        for (&dof, target) in work.moved.iter().zip(targets_at(work, &started, steps, DT)) {
            assert_eq!(found_x[dof].to_bits(), target.to_bits());
            assert_eq!(last_x[dof].to_bits(), target.to_bits());
        }
    }
    Ok(())
}

#[test]
fn a_step_on_the_device_is_the_step_of_the_integrator_to_the_bit() -> Result<(), AssertionError> {
    let work = steelwork();
    let Some(mut resident) = on_device(&work.model, false) else {
        return Ok(());
    };
    let masses = Lumped.masses(&work.model);
    let per_node = masses.iter().map(|mass| mass.value()).collect::<Vec<_>>();
    let imposed = Imposed {
        fixed: work.fixed.clone(),
        ..Imposed::default()
    };
    resident
        .start_dynamics(
            &masses,
            &deformed(&work.coordinates),
            &work.velocities(),
            &imposed,
            &(),
            &(),
        )
        .map_err(describe)?;
    for step in 0..10 {
        let (x, v, a) = resident.dynamics_state().map_err(describe)?;
        let (x, v, a) = (flat(&x), flat(&v), flat(&a));
        resident.steps(DT, 1, &(), &()).map_err(describe)?;
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
                if work.fixed.contains(&i) {
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
fn a_step_with_loads_and_moved_coordinates_is_the_step_of_the_integrator_to_the_bit()
-> Result<(), AssertionError> {
    let work = steelwork();
    let loads = loads_on(&work.model, &work.coordinates)?;
    let forces = device_forces_of(&loads);
    let imposed = imposed_by(&work, true);
    let Some(mut resident) = on_device(&work.model, false) else {
        return Ok(());
    };
    let masses = Lumped.masses(&work.model);
    let per_node = masses.iter().map(|mass| mass.value()).collect::<Vec<_>>();
    let start = deformed(&work.coordinates);
    let started = flat(&start);
    let schedule = schedule_of(&work, &start, DT);
    resident
        .start_dynamics(
            &masses,
            &start,
            &work.velocities(),
            &imposed,
            &loads,
            &schedule,
        )
        .map_err(describe)?;
    for step in 0..10 {
        let row = step + 1;
        let (x, v, a) = resident.dynamics_state().map_err(describe)?;
        let (x, v, a) = (flat(&x), flat(&v), flat(&a));
        resident.steps(DT, 1, &loads, &schedule).map_err(describe)?;
        let (x_next, v_next, a_next) = resident.dynamics_state().map_err(describe)?;
        resident.run().map_err(describe)?;
        let mut unbalanced = components_of(&resident)?;
        let mut which = 0;
        for force in &forces {
            match force {
                DeviceForce::Constant(values) => unbalanced
                    .iter_mut()
                    .zip(values)
                    .for_each(|(unbalanced, force)| *unbalanced -= force),
                DeviceForce::Weighted(weights) => {
                    let of_row = loads.device_components(which, time_at(row, DT));
                    which += 1;
                    for (node, weight) in weights.iter().enumerate() {
                        for (k, component) in of_row.iter().enumerate() {
                            unbalanced[3 * node + k] -= weight * component;
                        }
                    }
                }
            }
        }
        let held = |dof: &usize| work.fixed.contains(dof) || work.moved.contains(dof);
        let half = (0..x.len())
            .map(|i| a[i] * (DT * 0.5) + v[i])
            .collect::<Vec<_>>();
        let mut expected_x = (0..x.len())
            .map(|i| half[i] * DT + x[i])
            .collect::<Vec<_>>();
        for (&dof, target) in work.moved.iter().zip(targets_at(&work, &started, row, DT)) {
            expected_x[dof] = target;
        }
        let expected_a = (0..x.len())
            .map(|i| {
                if held(&i) {
                    0.0
                } else {
                    -unbalanced[i] / per_node[i / 3]
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

fn components_of(resident: &Resident) -> Result<Vec<f64>, AssertionError> {
    Ok(components(&resident.download().map_err(describe)?))
}

#[test]
fn steps_on_the_device_are_those_of_the_integrator_of_the_host() -> Result<(), AssertionError> {
    against_the_host(&steelwork(), &(), false)
}

#[test]
fn steps_with_loads_and_moved_coordinates_are_those_of_the_integrator_of_the_host()
-> Result<(), AssertionError> {
    let work = steelwork();
    against_the_host(&work, &loads_on(&work.model, &work.coordinates)?, true)
}

#[test]
fn blocks_of_both_kinds_take_the_steps_of_the_integrator_of_the_host_with_loads_and_motion()
-> Result<(), AssertionError> {
    let work = both_kinds();
    against_the_host(&work, &loads_on(&work.model, &work.coordinates)?, true)
}

#[test]
fn steps_taken_together_are_those_taken_one_at_a_time() -> Result<(), AssertionError> {
    let work = steelwork();
    let Some(mut resident) = on_device(&work.model, true) else {
        return Ok(());
    };
    let loads = loads_on(&work.model, &work.coordinates)?;
    let imposed = imposed_by(&work, true);
    let masses = Lumped.masses(&work.model);
    let (start, velocities) = (deformed(&work.coordinates), work.velocities());
    let schedule = schedule_of(&work, &start, DT);
    let mut run = |calls: usize, steps: usize| -> Result<[Vec<u64>; 3], AssertionError> {
        resident
            .start_dynamics(&masses, &start, &velocities, &imposed, &loads, &schedule)
            .map_err(describe)?;
        for _ in 0..calls {
            resident
                .steps(DT, steps, &loads, &schedule)
                .map_err(describe)?;
        }
        state(&resident)
    };
    let together = run(1, 1500)?;
    assert_eq!(run(15, 100)?, together);
    assert_eq!(run(1500, 1)?, together);
    Ok(())
}

#[test]
fn an_element_with_an_error_in_a_run_of_dynamics_is_an_error() -> Result<(), AssertionError> {
    let work = steelwork();
    let Some(mut resident) = on_device(&work.model, false) else {
        return Ok(());
    };
    let masses = Lumped.masses(&work.model);
    let velocities = work.velocities();
    let good = deformed(&work.coordinates);
    let mut bad = good.clone();
    bad[1] = [0.9, 0.9, 0.9].into();
    let imposed = Imposed {
        fixed: work.fixed.clone(),
        ..Imposed::default()
    };
    let error = resident
        .start_dynamics(&masses, &bad, &velocities, &imposed, &(), &())
        .expect_err("the start has an error");
    assert!(matches!(error, GpuError::Element { .. }), "{error}");
    resident
        .start_dynamics(&masses, &good, &velocities, &imposed, &(), &())
        .map_err(describe)?;
    resident.steps(DT, 2, &(), &()).map_err(describe)?;
    resident.set_coordinates(&bad).map_err(describe)?;
    let error = resident
        .steps(DT, 1, &(), &())
        .expect_err("the step has an error");
    assert!(matches!(error, GpuError::Element { .. }), "{error}");
    Ok(())
}

#[test]
fn what_is_not_a_run_of_the_model_is_an_error() -> Result<(), AssertionError> {
    let work = steelwork();
    let Some(mut resident) = on_device(&work.model, false) else {
        return Ok(());
    };
    let error = resident
        .steps(DT, 1, &(), &())
        .expect_err("a run that is not started is stepped");
    assert!(matches!(error, GpuError::Invalid(_)), "{error}");
    let error = resident
        .dynamics_state()
        .expect_err("the state of a run that is not started");
    assert!(matches!(error, GpuError::Invalid(_)), "{error}");
    let masses = Lumped.masses(&work.model);
    let (start, velocities) = (deformed(&work.coordinates), work.velocities());
    let grid = schedule_of(&work, &start, DT);
    let moved = Imposed {
        moved: work.moved.clone(),
        ..Imposed::default()
    };
    let error = resident
        .start_dynamics(&masses, &start, &velocities, &moved, &(), &())
        .expect_err("coordinates are moved and nothing says where");
    assert!(matches!(error, GpuError::Invalid(_)), "{error}");
    let outside = Imposed {
        fixed: vec![3 * work.coordinates.len()],
        ..Imposed::default()
    };
    let error = resident
        .start_dynamics(&masses, &start, &velocities, &outside, &(), &())
        .expect_err("a degree of freedom that the model does not have is held");
    assert!(matches!(error, GpuError::Invalid(_)), "{error}");
    let other = steelwork_of(COLUMNS - 1, LAYERS - 1);
    let others = loads_on(&other.model, &other.coordinates)?;
    let error = resident
        .start_dynamics(
            &masses,
            &start,
            &velocities,
            &Imposed::default(),
            &others,
            &grid,
        )
        .expect_err("loads of another number of nodes");
    assert!(matches!(error, GpuError::Invalid(_)), "{error}");
    let loads = loads_on(&work.model, &work.coordinates)?;
    let error = resident
        .start_dynamics(
            &masses,
            &start,
            &velocities,
            &Imposed::default(),
            &loads,
            &(),
        )
        .expect_err("loads of time are run and nothing says when");
    assert!(matches!(error, GpuError::Invalid(_)), "{error}");
    resident
        .start_dynamics(&masses, &start, &velocities, &Imposed::default(), &(), &())
        .map_err(describe)?;
    let error = resident
        .steps(DT, 1, &loads, &grid)
        .expect_err("the loads are not those of the run");
    assert!(matches!(error, GpuError::Invalid(_)), "{error}");
    Ok(())
}

#[test]
fn the_blocks_of_the_work_are_those_that_a_device_takes() {
    fn takes<B: DeviceBlocks>(_: &Model<B, 3>) {}
    takes(&steelwork().model);
    takes(&both_kinds().model);
}

fn subtract_as_the_host_does<B>(
    work: &Work<B>,
    loads: &(impl LoadAt<NodalCoordinates<3>, NodalForcesSolid<3>> + DeviceLoads),
) -> Result<(), AssertionError>
where
    B: ElasticElements<3>,
{
    let current = deformed(&work.coordinates);
    let forces = device_forces_of(loads);
    let [
        DeviceForce::Constant(_),
        DeviceForce::Weighted(_),
        DeviceForce::Constant(_),
    ] = forces.as_slice()
    else {
        return Err("the loads are a constant force, a force of time, and a constant force".into());
    };
    assert_eq!(loads.device_weighted(), 1);
    let start = work.model.nodal_forces(&current)?;
    for row in [0, 1, 7, 100] {
        let mut expected = start.clone();
        loads.subtract_at(time_at(row, DT), &current, &mut expected)?;
        let mut found = components(&start);
        let mut which = 0;
        for force in &forces {
            match force {
                DeviceForce::Constant(values) => found
                    .iter_mut()
                    .zip(values)
                    .for_each(|(found, value)| *found -= value),
                DeviceForce::Weighted(weights) => {
                    let of_time = loads.device_components(which, time_at(row, DT));
                    which += 1;
                    for (node, weight) in weights.iter().enumerate() {
                        for (k, component) in of_time.iter().enumerate() {
                            found[3 * node + k] -= weight * component;
                        }
                    }
                }
            }
        }
        assert_eq!(bits(&found), bits(&components(&expected)), "row {row}");
    }
    Ok(())
}

#[test]
fn the_loads_that_a_device_takes_are_the_forces_that_the_host_subtracts()
-> Result<(), AssertionError> {
    let work = steelwork();
    let loads = loads_on(&work.model, &work.coordinates)?;
    let forces = device_forces_of(&loads);
    let [DeviceForce::Constant(first), _, DeviceForce::Constant(last)] = forces.as_slice() else {
        return Err("the loads are a constant force, a force of time, and a constant force".into());
    };
    assert_eq!(
        bits(first),
        bits(&components(&static_forces(&work.coordinates, FIRST)))
    );
    assert_eq!(
        bits(last),
        bits(&components(&static_forces(&work.coordinates, LAST)))
    );
    subtract_as_the_host_does(&work, &loads)
}

#[test]
fn the_loads_on_the_volume_are_subtracted_as_the_host_subtracts_them() -> Result<(), AssertionError>
{
    let work = steelwork();
    subtract_as_the_host_does(&work, &volume_loads_on(&work)?)
}

#[test]
fn no_loads_are_no_forces_and_a_load_of_time_is_one_of_those_of_time() -> Result<(), AssertionError>
{
    assert!(device_forces_of(&()).is_empty());
    assert_eq!(().device_weighted(), 0);
    let work = steelwork();
    let loads = loads_on(&work.model, &work.coordinates)?;
    assert_eq!(device_forces_of(&loads).len(), 3);
    assert_eq!(loads.device_weighted(), 1);
    Ok(())
}
