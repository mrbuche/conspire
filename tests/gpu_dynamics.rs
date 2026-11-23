#![cfg(all(feature = "gpu", feature = "fem"))]

mod common;

use common::{
    FIRST, LAST, RATE, Work, bits, close, components, deformed, density, describe, flat,
    imposed_by, lattice, loads_on, on_device, schedule_of, slab, static_forces, steel, targets_at,
    time_at, volume_loads_on,
};
use conspire::{
    constitutive::solid::hyperelastic::SaintVenantKirchhoff,
    fem::{
        Blocks, Model, NodalAccelerations, NodalCoordinates, NodalReferenceCoordinates,
        NodalVelocities,
        block::Block,
        boundary::{BoundaryConditions, ConstrainAt, DeviceForce, Displacement, LoadAt, Solid},
        element::{linear::Hexahedron, stabilized::linear::Hexahedron as StabilizedHexahedron},
        solid::{
            ElasticExplicitDynamics, Execution, Lumped, MassChoice, NodalForcesSolid, Unchecked,
            elastic::ElasticElements,
        },
    },
    gpu::{DeviceBlocks, GpuError, Imposed, ImposedLoad, Resident, Schedule},
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

// A step that a time grid of these has is exact in binary, so that the steps of the integrator of
// the host, which takes each from the difference of times, are all this.
const DT: f64 = 1.0 / 1_048_576.0;

fn steelwork() -> Work<Steel> {
    let coordinates = lattice(COLUMNS, LAYERS);
    let reference = NodalReferenceCoordinates::from(coordinates.clone());
    let block = Steel::from((steel(), density(), slab(COLUMNS, 0..LAYERS), &reference));
    Work::new((block, reference).into(), coordinates, COLUMNS)
}

// Blocks of both kinds that share nodes.
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

// A run on the device is a run of the integrator of the library: the same steps from the same
// state, with the same loads and the same motion, end where it ends, to the precision of the
// forces, which are made in another order of operations on the device. The coordinates that are
// moved are where they are held, exactly.
fn against_the_host<B>(
    work: &Work<B>,
    loads: &impl LoadAt<NodalCoordinates<3>, NodalForcesSolid<3>>,
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
    let forces = loads
        .device_forces()
        .ok_or("the loads are not ones that a device takes")?;
    let (imposed, components) = imposed_by(work, forces, moved, DT);
    let Some(mut resident) = on_device(&work.model, false) else {
        return Ok(());
    };
    let targets = |row| targets_at(work, &flat(&start), row, DT);
    let schedule = schedule_of(
        moved.then_some(&targets as &dyn Fn(usize) -> Vec<f64>),
        &components,
    );
    resident
        .start_dynamics(
            &Lumped.masses(&work.model),
            &start,
            &velocities,
            &imposed,
            &schedule,
        )
        .map_err(describe)?;
    resident.steps(DT, steps, &schedule).map_err(describe)?;
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
    // The fixed coordinates are where they were, and the moved ones are where they are held, as
    // the host has them.
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

// The first half of a step and the second, which the device does with its own kernels, are the
// operations of the integrator of the library, in the same order, so that from the state and the
// forces that the device has, the host makes the next state to the bit, whether or not the
// forces are the same as the host's: each step is checked against the forces that the device
// itself made.
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
            &Schedule::default(),
        )
        .map_err(describe)?;
    for step in 0..10 {
        let (x, v, a) = resident.dynamics_state().map_err(describe)?;
        let (x, v, a) = (flat(&x), flat(&v), flat(&a));
        resident
            .steps(DT, 1, &Schedule::default())
            .map_err(describe)?;
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

// The same, with the loads subtracted from the forces in their order and the coordinates that
// are moved set to where they are held, which are the other operations of a step.
#[test]
fn a_step_with_loads_and_moved_coordinates_is_the_step_of_the_integrator_to_the_bit()
-> Result<(), AssertionError> {
    let work = steelwork();
    let loads = loads_on(&work.model, &work.coordinates)?;
    let forces = loads
        .device_forces()
        .ok_or("the loads are not ones that a device takes")?;
    let (imposed, components) = imposed_by(&work, forces, true, DT);
    let Some(mut resident) = on_device(&work.model, false) else {
        return Ok(());
    };
    let masses = Lumped.masses(&work.model);
    let per_node = masses.iter().map(|mass| mass.value()).collect::<Vec<_>>();
    let start = deformed(&work.coordinates);
    let targets = |row| targets_at(&work, &flat(&start), row, DT);
    let schedule = schedule_of(Some(&targets), &components);
    resident
        .start_dynamics(&masses, &start, &work.velocities(), &imposed, &schedule)
        .map_err(describe)?;
    for step in 0..10 {
        let row = step + 1;
        let (x, v, a) = resident.dynamics_state().map_err(describe)?;
        let (x, v, a) = (flat(&x), flat(&v), flat(&a));
        resident.steps(DT, 1, &schedule).map_err(describe)?;
        let (x_next, v_next, a_next) = resident.dynamics_state().map_err(describe)?;
        // The forces of the elements at the coordinates that the step made, less the loads.
        resident.run().map_err(describe)?;
        let mut unbalanced = components_of(&resident)?;
        let mut weighted = components.iter();
        for load in &imposed.loads {
            match load {
                ImposedLoad::Constant(forces) => unbalanced
                    .iter_mut()
                    .zip(forces)
                    .for_each(|(unbalanced, force)| *unbalanced -= force),
                ImposedLoad::Weighted(weights) => {
                    let of_row = weighted.next().ok_or("the components of the load")?(row);
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
        for (&dof, target) in work.moved.iter().zip(targets(row)) {
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

// The forces of the elements that the device has made last.
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

// The same with blocks of both kinds, which share nodes.
#[test]
fn blocks_of_both_kinds_take_the_steps_of_the_integrator_of_the_host_with_loads_and_motion()
-> Result<(), AssertionError> {
    let work = both_kinds();
    against_the_host(&work, &loads_on(&work.model, &work.coordinates)?, true)
}

// The steps that are taken in one call are those taken in many, and those that go through more
// rows of the tables than a device holds at a time are the same: nothing is kept between the
// calls but the state on the device.
#[test]
fn steps_taken_together_are_those_taken_one_at_a_time() -> Result<(), AssertionError> {
    let work = steelwork();
    let Some(mut resident) = on_device(&work.model, true) else {
        return Ok(());
    };
    let loads = loads_on(&work.model, &work.coordinates)?;
    let forces = loads
        .device_forces()
        .ok_or("the loads are not ones that a device takes")?;
    let (imposed, components) = imposed_by(&work, forces, true, DT);
    let masses = Lumped.masses(&work.model);
    let (start, velocities) = (deformed(&work.coordinates), work.velocities());
    let targets = |row| targets_at(&work, &flat(&start), row, DT);
    let schedule = schedule_of(Some(&targets), &components);
    let mut run = |calls: usize, steps: usize| -> Result<[Vec<u64>; 3], AssertionError> {
        resident
            .start_dynamics(&masses, &start, &velocities, &imposed, &schedule)
            .map_err(describe)?;
        for _ in 0..calls {
            resident.steps(DT, steps, &schedule).map_err(describe)?;
        }
        state(&resident)
    };
    // More steps than a table has rows, which is 1024 here, in one call, in fifteen, and singly.
    let together = run(1, 1500)?;
    assert_eq!(run(15, 100)?, together);
    assert_eq!(run(1500, 1)?, together);
    Ok(())
}

// The state that is put into nodes that are kept is the state that comes in new nodes, to the
// bit, whatever the kept nodes held, call after call.
#[test]
fn the_state_put_into_nodes_that_are_kept_is_the_state_in_new_nodes_to_the_bit()
-> Result<(), AssertionError> {
    let work = steelwork();
    let Some(mut resident) = on_device(&work.model, true) else {
        return Ok(());
    };
    let imposed = Imposed {
        fixed: work.fixed.clone(),
        ..Imposed::default()
    };
    let schedule = Schedule::default();
    resident
        .start_dynamics(
            &Lumped.masses(&work.model),
            &deformed(&work.coordinates),
            &work.velocities(),
            &imposed,
            &schedule,
        )
        .map_err(describe)?;
    let nodes = work.coordinates.len();
    let mut kept = (
        NodalCoordinates::zero(nodes),
        NodalVelocities::zero(nodes),
        NodalAccelerations::zero(nodes),
    );
    for vector in [
        kept.0.as_flat_mut(),
        kept.1.as_flat_mut(),
        kept.2.as_flat_mut(),
    ] {
        vector.fill(f64::NAN);
    }
    for _ in 0..4 {
        resident.steps(DT, 3, &schedule).map_err(describe)?;
        let new = resident.dynamics_state().map_err(describe)?;
        resident
            .dynamics_state_into(&mut kept.0, &mut kept.1, &mut kept.2)
            .map_err(describe)?;
        assert_eq!(bits(kept.0.as_flat()), bits(new.0.as_flat()));
        assert_eq!(bits(kept.1.as_flat()), bits(new.1.as_flat()));
        assert_eq!(bits(kept.2.as_flat()), bits(new.2.as_flat()));
    }
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
    // A node of the bottom layer moved through the elements that have it.
    bad[1] = [0.9, 0.9, 0.9].into();
    let imposed = Imposed {
        fixed: work.fixed.clone(),
        ..Imposed::default()
    };
    let schedule = Schedule::default();
    let error = resident
        .start_dynamics(&masses, &bad, &velocities, &imposed, &schedule)
        .expect_err("the start has an error");
    assert!(matches!(error, GpuError::Element { .. }), "{error}");
    resident
        .start_dynamics(&masses, &good, &velocities, &imposed, &schedule)
        .map_err(describe)?;
    resident.steps(DT, 2, &schedule).map_err(describe)?;
    // The coordinates are put where an element has an error, as a step that does that would.
    resident.set_coordinates(&bad).map_err(describe)?;
    let error = resident
        .steps(DT, 1, &schedule)
        .expect_err("the step has an error");
    assert!(matches!(error, GpuError::Element { .. }), "{error}");
    Ok(())
}

// What is not a run of the model, or of what is imposed, is an error that says so. What each
// check says is tested without a device; this is that a run is not started on one.
#[test]
fn what_is_not_a_run_of_the_model_is_an_error() -> Result<(), AssertionError> {
    let work = steelwork();
    let Some(mut resident) = on_device(&work.model, false) else {
        return Ok(());
    };
    let schedule = Schedule::default();
    let error = resident
        .steps(DT, 1, &schedule)
        .expect_err("a run that is not started is stepped");
    assert!(matches!(error, GpuError::Invalid(_)), "{error}");
    let error = resident
        .dynamics_state()
        .expect_err("the state of a run that is not started");
    assert!(matches!(error, GpuError::Invalid(_)), "{error}");
    let masses = Lumped.masses(&work.model);
    let (start, velocities) = (deformed(&work.coordinates), work.velocities());
    let mut start_with = |imposed: Imposed, schedule: &Schedule<'_>| {
        resident.start_dynamics(&masses, &start, &velocities, &imposed, schedule)
    };
    // Coordinates that are moved, and nothing to say where.
    let error = start_with(
        Imposed {
            moved: work.moved.clone(),
            ..Imposed::default()
        },
        &schedule,
    )
    .expect_err("coordinates are moved and nothing says where");
    assert!(matches!(error, GpuError::Invalid(_)), "{error}");
    // A degree of freedom that the model does not have.
    let error = start_with(
        Imposed {
            fixed: vec![3 * work.coordinates.len()],
            ..Imposed::default()
        },
        &schedule,
    )
    .expect_err("a degree of freedom that the model does not have is held");
    assert!(matches!(error, GpuError::Invalid(_)), "{error}");
    // Loads of another number of nodes.
    let error = start_with(
        Imposed {
            loads: vec![ImposedLoad::Constant(vec![0.0; 5])],
            ..Imposed::default()
        },
        &schedule,
    )
    .expect_err("a load of another number of nodes");
    assert!(matches!(error, GpuError::Invalid(_)), "{error}");
    Ok(())
}

#[test]
fn the_blocks_of_the_work_are_those_that_a_device_takes() {
    fn takes<B: DeviceBlocks>(_: &Model<B, 3>) {}
    takes(&steelwork().model);
    takes(&both_kinds().model);
}

// The loads that a device takes subtract the forces that the host subtracts, in the order that it
// does, to the bit, whatever they are; and they are a constant force, a force of time, and a
// constant force, in that order.
fn subtract_as_the_host_does<B>(
    work: &Work<B>,
    loads: &impl LoadAt<NodalCoordinates<3>, NodalForcesSolid<3>>,
) -> Result<(), AssertionError>
where
    B: ElasticElements<3>,
{
    let current = deformed(&work.coordinates);
    let forces = loads
        .device_forces()
        .ok_or("the loads are not ones that a device takes")?;
    let [
        DeviceForce::Constant(_),
        DeviceForce::Weighted { .. },
        DeviceForce::Constant(_),
    ] = forces.as_slice()
    else {
        return Err("the loads are a constant force, a force of time, and a constant force".into());
    };
    let start = work.model.nodal_forces(&current)?;
    for row in [0, 1, 7, 100] {
        let mut expected = start.clone();
        loads.subtract_at(time_at(row, DT), &current, &mut expected)?;
        let mut found = components(&start);
        for force in &forces {
            match force {
                DeviceForce::Constant(values) => found
                    .iter_mut()
                    .zip(values)
                    .for_each(|(found, value)| *found -= value),
                DeviceForce::Weighted {
                    weights,
                    components,
                } => {
                    let of_time = components(time_at(row, DT));
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
    // The first and the last are the forces that the loads were given, in that order.
    let forces = loads
        .device_forces()
        .ok_or("the loads are not ones that a device takes")?;
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

// The loads on the volume that the benchmark has are loads that a device takes in the same way.
#[test]
fn the_loads_on_the_volume_are_subtracted_as_the_host_subtracts_them() -> Result<(), AssertionError>
{
    let work = steelwork();
    subtract_as_the_host_does(&work, &volume_loads_on(&work)?)
}

#[test]
fn a_load_that_follows_the_body_is_not_one_that_a_device_takes() -> Result<(), AssertionError> {
    struct Pressure;
    impl LoadAt<NodalCoordinates<3>, NodalForcesSolid<3>> for Pressure {
        fn subtract_at(
            &self,
            _: Quantity<Time>,
            _: &NodalCoordinates<3>,
            _: &mut NodalForcesSolid<3>,
        ) -> Result<(), String> {
            Ok(())
        }
    }
    let work = steelwork();
    let loads = loads_on(&work.model, &work.coordinates)?;
    assert!(Pressure.device_forces().is_none());
    assert!(
        (loads_on(&work.model, &work.coordinates)?, Pressure)
            .device_forces()
            .is_none()
    );
    assert!((Pressure, loads).device_forces().is_none());
    let none = LoadAt::<NodalCoordinates<3>, NodalForcesSolid<3>>::device_forces(&());
    assert!(none.ok_or("no loads are none")?.is_empty());
    Ok(())
}
