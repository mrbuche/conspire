#![cfg(all(feature = "gpu", feature = "fem"))]

mod common;

use common::{
    FIRST, LAST, Work, bits, close, components, conditions_on, deformed, density, describe,
    device_forces_of, flat, lattice, loads_on, on_device, slab, static_forces, steel, targets_at,
    time_at, volume_loads_on,
};
use conspire::{
    constitutive::solid::hyperelastic::SaintVenantKirchhoff,
    fem::{
        Blocks, Model, NodalAccelerations, NodalCoordinates, NodalReferenceCoordinates,
        NodalVelocities,
        block::Block,
        boundary::{
            BoundaryConditions, ConstrainAt, DeviceForce, DeviceLoads, Displacement, Enforcement,
            LoadAt, Solid,
        },
        element::{linear::Hexahedron, stabilized::linear::Hexahedron as StabilizedHexahedron},
        solid::{
            ElasticExplicitDynamics, Execution, Lumped, MassChoice, NodalForcesSolid, Unchecked,
            elastic::ElasticElements,
        },
    },
    gpu::{DeviceBlocks, ExplicitRun, GpuError, Resident},
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

fn start<'a, B, C, L>(
    resident: &'a mut Resident,
    work: &Work<B>,
    loads: &'a L,
    conditions: &'a C,
) -> Result<ExplicitRun<'a, C, L>, GpuError>
where
    Lumped: MassChoice<Model<B, 3>, Mass = QuantityVector<Mass>>,
    C: ConstrainAt<Solid<3>>,
    L: DeviceLoads,
{
    resident.explicit_dynamics(
        &Lumped.masses(&work.model),
        &VelocityVerlet::new(Time::seconds(DT)),
        Time::seconds(0.0),
        (&deformed(&work.coordinates), &work.velocities()),
        loads,
        conditions,
    )
}

fn state<C, L>(run: &ExplicitRun<'_, C, L>) -> Result<[Vec<u64>; 3], AssertionError> {
    let (x, v, a) = run.state().map_err(describe)?;
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
    let (initial, velocities) = (deformed(&work.coordinates), work.velocities());
    let conditions = conditions_on(work, moved);
    let time = (0..=steps).map(|row| time_at(row, DT)).collect::<Vec<_>>();
    let (_, x, v, a) = work.model.integrate(
        Execution::Serial,
        Unchecked,
        Lumped,
        &VelocityVerlet::new(Time::seconds(DT)),
        &time,
        (initial.clone(), velocities),
        loads,
        &conditions,
    )?;
    let Some(mut resident) = on_device(&work.model, false) else {
        return Ok(());
    };
    let mut run = start(&mut resident, work, loads, &conditions).map_err(describe)?;
    run.steps(steps).map_err(describe)?;
    let (found_x, found_v, found_a) = run.state().map_err(describe)?;
    let last = |what: &str| format!("the host's run has {what}");
    let (last_x, last_v, last_a) = (
        flat(x.iter().last().ok_or_else(|| last("coordinates"))?),
        flat(v.iter().last().ok_or_else(|| last("velocities"))?),
        flat(a.iter().last().ok_or_else(|| last("accelerations"))?),
    );
    close(&flat(&found_x), &last_x, "coordinates")?;
    close(&flat(&found_v), &last_v, "velocities")?;
    close(&flat(&found_a), &last_a, "accelerations")?;
    let (found_x, started) = (flat(&found_x), flat(&initial));
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
    let conditions = conditions_on(&work, false);
    let mut run = start(&mut resident, &work, &(), &conditions).map_err(describe)?;
    for step in 0..10 {
        let (x, v, a) = run.state().map_err(describe)?;
        let (x, v, a) = (flat(&x), flat(&v), flat(&a));
        run.steps(1).map_err(describe)?;
        let (x_next, v_next, a_next) = run.state().map_err(describe)?;
        let forces = components(&run.forces().map_err(describe)?);
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
    let conditions = conditions_on(&work, true);
    let Some(mut resident) = on_device(&work.model, false) else {
        return Ok(());
    };
    let masses = Lumped.masses(&work.model);
    let per_node = masses.iter().map(|mass| mass.value()).collect::<Vec<_>>();
    let started = flat(&deformed(&work.coordinates));
    let mut run = start(&mut resident, &work, &loads, &conditions).map_err(describe)?;
    for step in 0..10 {
        let row = step + 1;
        let (x, v, a) = run.state().map_err(describe)?;
        let (x, v, a) = (flat(&x), flat(&v), flat(&a));
        run.steps(1).map_err(describe)?;
        let (x_next, v_next, a_next) = run.state().map_err(describe)?;
        let mut unbalanced = components(&run.forces().map_err(describe)?);
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
    let conditions = conditions_on(&work, true);
    let mut taken = |calls: usize, steps: usize| -> Result<[Vec<u64>; 3], AssertionError> {
        let mut run = start(&mut resident, &work, &loads, &conditions).map_err(describe)?;
        for _ in 0..calls {
            run.steps(steps).map_err(describe)?;
        }
        state(&run)
    };
    let together = taken(1, 1500)?;
    assert_eq!(taken(15, 100)?, together);
    assert_eq!(taken(1500, 1)?, together);
    Ok(())
}

#[test]
fn the_state_put_into_nodes_that_are_kept_is_the_state_in_new_nodes_to_the_bit()
-> Result<(), AssertionError> {
    let work = steelwork();
    let Some(mut resident) = on_device(&work.model, true) else {
        return Ok(());
    };
    let conditions = conditions_on(&work, false);
    let mut run = start(&mut resident, &work, &(), &conditions).map_err(describe)?;
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
        run.steps(3).map_err(describe)?;
        let new = run.state().map_err(describe)?;
        run.state_into(&mut kept.0, &mut kept.1, &mut kept.2)
            .map_err(describe)?;
        assert_eq!(bits(kept.0.as_flat()), bits(new.0.as_flat()));
        assert_eq!(bits(kept.1.as_flat()), bits(new.1.as_flat()));
        assert_eq!(bits(kept.2.as_flat()), bits(new.2.as_flat()));
    }
    Ok(())
}

fn refused<T>(run: Result<T, GpuError>) -> Result<GpuError, AssertionError> {
    run.err().ok_or_else(|| "the run is started".into())
}

#[test]
fn an_element_with_an_error_in_a_run_of_dynamics_is_an_error() -> Result<(), AssertionError> {
    let work = steelwork();
    let Some(mut resident) = on_device(&work.model, false) else {
        return Ok(());
    };
    let masses = Lumped.masses(&work.model);
    let integrator = VelocityVerlet::new(Time::seconds(DT));
    let conditions = conditions_on(&work, false);
    let mut bad = deformed(&work.coordinates);
    bad[1] = [0.9, 0.9, 0.9].into();
    let error = refused(resident.explicit_dynamics(
        &masses,
        &integrator,
        Time::seconds(0.0),
        (&bad, &work.velocities()),
        &(),
        &conditions,
    ))?;
    assert!(matches!(error, GpuError::Element { .. }), "{error}");
    let mut velocities = work.velocities();
    let last = work.coordinates.len() - 1;
    velocities[last] = [-2.0 / DT; 3].into();
    let mut run = resident
        .explicit_dynamics(
            &masses,
            &integrator,
            Time::seconds(0.0),
            (&deformed(&work.coordinates), &velocities),
            &(),
            &conditions,
        )
        .map_err(describe)?;
    let error = refused(run.steps(1))?;
    assert!(matches!(error, GpuError::Element { .. }), "{error}");
    Ok(())
}

#[test]
fn what_is_not_a_run_of_the_model_is_an_error() -> Result<(), AssertionError> {
    let work = steelwork();
    let Some(mut resident) = on_device(&work.model, false) else {
        return Ok(());
    };
    let outside =
        BoundaryConditions::<Solid<3>>::new().fix_node(Displacement::<0>, work.coordinates.len());
    let error = refused(start(&mut resident, &work, &(), &outside))?;
    assert!(matches!(error, GpuError::Invalid(_)), "{error}");
    let multipliers = BoundaryConditions::<Solid<3>>::new().prescribe_nodes_by(
        Enforcement::Multipliers,
        Displacement::<0>,
        [0],
        Length::meters(0.0),
    );
    let error = refused(start(&mut resident, &work, &(), &multipliers))?;
    assert!(matches!(error, GpuError::Invalid(_)), "{error}");
    let conditions = conditions_on(&work, false);
    let other = steelwork_of(COLUMNS - 1, LAYERS - 1);
    let others = loads_on(&other.model, &other.coordinates)?;
    let error = refused(start(&mut resident, &work, &others, &conditions))?;
    assert!(matches!(error, GpuError::Invalid(_)), "{error}");
    let error = refused(resident.explicit_dynamics(
        &Lumped.masses(&work.model),
        &VelocityVerlet::new(Time::seconds(0.0)),
        Time::seconds(0.0),
        (&deformed(&work.coordinates), &work.velocities()),
        &(),
        &conditions,
    ))?;
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
