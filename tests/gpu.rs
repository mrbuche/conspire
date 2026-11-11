#![cfg(all(feature = "gpu", feature = "fem"))]

//! The blocks of a model as a device takes them, which need no device to check: the second pass
//! of the forces, which a device does by adding the forces of the elements to each node in the
//! order of the elements, is done here on the host from the layout that a device is given, and
//! must give the nodal forces of the model to the bit. And the forces on a device, which need
//! one.

mod common;

use common::{
    DT, RATE, Work, agree, bits, close, components, deformed, density, disjoint, flat, imposed_by,
    lattice, loads_on, on_device, rubber, schedule_of, slab, steel, targets_at, time_of,
};
use conspire::{
    constitutive::solid::{
        elastic::{DeviceModel, Elastic},
        hyperelastic::{NeoHookean, SaintVenantKirchhoff},
    },
    fem::{
        Blocks, Model, NodalAccelerations, NodalCoordinates, NodalReferenceCoordinates,
        NodalVelocities,
        block::Block,
        boundary::{BoundaryConditions, ConstrainAt, DeviceForce, Displacement, LoadAt, Solid},
        element::{
            ElementNodalCoordinates, ElementNodalReferenceCoordinates, FiniteElementError,
            OnTheFly,
            linear::{Hexahedron, LinearElement, Tetrahedron},
            solid::{
                ElementNodalForcesSolid,
                elastic::{ElasticElement, ElasticFiniteElement},
            },
        },
        solid::{
            ElasticExplicitDynamics, Execution, Lumped, MassChoice, NodalForcesSolid, Unchecked,
            elastic::ElasticElements,
        },
    },
    gpu::{
        DeviceBlock, DeviceBlocks, DeviceError, GpuError, Imposed, ImposedLoad, Resident, Schedule,
    },
    math::{Quantity, Tensor, integrate::VelocityVerlet},
    units::{Density, Length, Time},
};

type Grid = Block<NeoHookean, Hexahedron, 8, 3, 8, 8, Quantity<Density>>;
type Steel = Block<SaintVenantKirchhoff, Hexahedron, 8, 3, 8, 8, Quantity<Density>>;
type Tetrahedra = Block<NeoHookean, Tetrahedron<1>, 1, 3, 4, 4, Quantity<Density>>;

const COLUMNS: usize = 3;
const LAYERS: usize = 4;

/// Two blocks, and what they were made of.
struct TwoBlocks {
    model: Model<Blocks<Grid, Steel>, 3>,
    coordinates: Vec<[f64; 3]>,
    below: Vec<[usize; 8]>,
    above: Vec<[usize; 8]>,
}

/// Two blocks that share the nodes of the layer between them: rubber below and steel above.
fn two_blocks() -> TwoBlocks {
    let coordinates = lattice(COLUMNS, LAYERS);
    let reference = NodalReferenceCoordinates::from(coordinates.clone());
    let (below, above) = (slab(COLUMNS, 0..2), slab(COLUMNS, 2..LAYERS));
    let blocks = Blocks(
        Grid::from((rubber(), density(), below.clone(), &reference)),
        Steel::from((steel(), density(), above.clone(), &reference)),
    );
    TwoBlocks {
        model: (blocks, reference).into(),
        coordinates,
        below,
        above,
    }
}

#[test]
fn a_block_is_described_by_its_model_its_shape_and_the_layout_of_its_elements() {
    let TwoBlocks {
        model,
        below,
        above,
        ..
    } = two_blocks();
    let mut blocks = Vec::new();
    model.device_blocks(&mut blocks);
    assert_eq!(blocks.len(), 2);
    let (tables_derivatives, tables_weights) = OnTheFly::<8, 8>::tables::<Hexahedron>();
    for (block, (name, parameters, connectivity)) in blocks.iter().zip([
        (NeoHookean::NAME, rubber().parameters(), &below),
        (SaintVenantKirchhoff::NAME, steel().parameters(), &above),
    ]) {
        assert_eq!(block.model, name);
        assert_eq!(block.parameters, parameters);
        assert_eq!(block.points, 8);
        assert_eq!(
            block.derivatives,
            tables_derivatives.as_flattened().as_flattened()
        );
        assert_eq!(block.weights, tables_weights);
        let items = connectivity.len();
        assert_eq!(block.layout.items, items);
        assert_eq!(block.layout.nodes_per_item, 8);
        connectivity.iter().enumerate().for_each(|(item, nodes)| {
            nodes.iter().enumerate().for_each(|(a, &node)| {
                assert_eq!(block.layout.connectivity[a * items + item], node as u32)
            })
        });
    }
    assert_eq!(blocks[0].parameters, [3.0e9, 13.0e9]);
    assert_eq!(blocks[1].parameters, [80.0e9, 160.0e9]);
}

#[test]
fn a_block_of_another_shape_is_described_the_same_way() {
    let coordinates = lattice(COLUMNS, LAYERS);
    let reference = NodalReferenceCoordinates::from(coordinates);
    // The six tetrahedra of a cube that share its diagonal.
    let connectivity = vec![
        [0, 1, 2, 6],
        [0, 2, 3, 6],
        [0, 3, 7, 6],
        [0, 7, 4, 6],
        [0, 4, 5, 6],
        [0, 5, 1, 6],
    ];
    let model: Model<Tetrahedra, 3> = (
        Tetrahedra::from((rubber(), density(), connectivity, &reference)),
        reference,
    )
        .into();
    let mut blocks = Vec::new();
    model.device_blocks(&mut blocks);
    assert_eq!(blocks.len(), 1);
    assert_eq!(blocks[0].model, NeoHookean::NAME);
    assert_eq!(blocks[0].points, 1);
    assert_eq!(blocks[0].layout.items, 6);
    assert_eq!(blocks[0].layout.nodes_per_item, 4);
    assert_eq!(blocks[0].derivatives.len(), 4 * 3);
    assert_eq!(blocks[0].weights.len(), 1);
}

/// The forces of the elements of a block, in the planes that a device makes them in: the
/// component `i` of the force on the node `a` of an element `item` is at `(i * 8 + a) * items +
/// item`.
fn element_forces<M>(
    block: &DeviceBlock,
    model: &M,
    connectivity: &[[usize; 8]],
    reference: &[[f64; 3]],
    current: &NodalCoordinates<3>,
) -> Vec<f64>
where
    M: Elastic,
    Hexahedron: ElasticElement<
            M,
            Forces = ElementNodalForcesSolid<8>,
            Coordinates = ElementNodalCoordinates<8>,
            Error = FiniteElementError,
        >,
{
    let items = block.layout.items;
    let mut forces = vec![0.0; 3 * block.layout.plane()];
    for (item, nodes) in connectivity.iter().enumerate() {
        let element = Hexahedron::from(ElementNodalReferenceCoordinates::<8>::from(
            core::array::from_fn::<[f64; 3], 8, _>(|a| reference[nodes[a]]),
        ));
        let computed = element
            .nodal_forces(
                model,
                &ElementNodalCoordinates::<8>::from(core::array::from_fn::<[f64; 3], 8, _>(|a| {
                    core::array::from_fn(|i| current[nodes[a]][i].value())
                })),
            )
            .expect("the forces of an element");
        for (a, force) in computed.iter().enumerate() {
            for (i, component) in force.iter().enumerate() {
                forces[(i * 8 + a) * items + item] = component.value();
            }
        }
    }
    forces
}

/// The second pass, as the kernel does it: each node adds the forces it is given in order.
fn gather(block: &DeviceBlock, forces: &[f64], nodal_forces: &mut [[f64; 3]]) {
    let layout = &block.layout;
    let plane = layout.plane();
    for (node, nodal_force) in nodal_forces
        .iter_mut()
        .enumerate()
        .take(layout.number_of_nodes())
    {
        let (first, last) = (
            layout.offsets[node] as usize,
            layout.offsets[node + 1] as usize,
        );
        let mut sum = *nodal_force;
        for &contribution in &layout.contributions[first..last] {
            sum.iter_mut()
                .enumerate()
                .for_each(|(i, value)| *value += forces[i * plane + contribution as usize]);
        }
        *nodal_force = sum;
    }
}

#[test]
fn the_second_pass_from_the_layout_gives_the_nodal_forces_of_the_model_to_the_bit() {
    let TwoBlocks {
        model,
        coordinates: reference,
        below,
        above,
    } = two_blocks();
    let current = deformed(&reference);
    let mut blocks = Vec::new();
    model.device_blocks(&mut blocks);
    let mut nodal_forces = vec![[0.0; 3]; reference.len()];
    let first = element_forces(&blocks[0], &rubber(), &below, &reference, &current);
    gather(&blocks[0], &first, &mut nodal_forces);
    let second = element_forces(&blocks[1], &steel(), &above, &reference, &current);
    gather(&blocks[1], &second, &mut nodal_forces);
    let expected = model.nodal_forces(&current).expect("the nodal forces");
    assert_eq!(expected.len(), nodal_forces.len());
    let mut largest = 0.0_f64;
    for (expected, found) in expected.iter().zip(&nodal_forces) {
        for (expected, found) in expected.iter().zip(found) {
            largest = largest.max(expected.value().abs());
            assert_eq!(
                expected.value().to_bits(),
                found.to_bits(),
                "{} and {found}",
                expected.value()
            );
        }
    }
    assert!(largest > 1e3, "the forces are {largest}, which is nothing");
}

// What follows runs on a device. Where there is none the tests say so and pass, unless
// `CONSPIRE_REQUIRE_GPU` is set, which makes a missing device or missing kernels a failure, so
// that a run on a machine that has a device cannot pass by skipping them.

#[test]
fn a_model_of_two_blocks_has_the_forces_of_the_host_on_a_device() {
    let TwoBlocks {
        model, coordinates, ..
    } = two_blocks();
    let current = deformed(&coordinates);
    let expected = components(
        &model
            .nodal_forces(&current)
            .expect("the forces on the host"),
    );
    for fused in [false, true] {
        let Some(resident) = on_device(&model, fused) else {
            return;
        };
        eprintln!("{}", resident.device_name());
        let found = resident
            .nodal_forces(&model, &current)
            .expect("the forces on the device");
        agree(&components(&found), &expected, 1e-10);
    }
}

/// The forces of the elements of the steel blocks as the host makes them from the data that the
/// kernels make on the fly: the same code.
fn on_the_fly_element_forces(
    block: &DeviceBlock,
    model: &SaintVenantKirchhoff,
    connectivity: &[[usize; 8]],
    reference: &[[f64; 3]],
    current: &NodalCoordinates<3>,
) -> Vec<f64> {
    let (derivatives, weights) = OnTheFly::<8, 8>::tables::<Hexahedron>();
    let items = block.layout.items;
    let mut forces = vec![0.0; 3 * block.layout.plane()];
    for (item, nodes) in connectivity.iter().enumerate() {
        let data = OnTheFly::new(
            core::array::from_fn::<[f64; 3], 8, _>(|a| reference[nodes[a]]),
            &derivatives,
            &weights,
        );
        let computed = <Hexahedron as ElasticFiniteElement<SaintVenantKirchhoff, 8, 3, 8, 8>>::nodal_forces_from(
            &data,
            model,
            &ElementNodalCoordinates::<8>::from(core::array::from_fn::<[f64; 3], 8, _>(|a| {
                core::array::from_fn(|i| current[nodes[a]][i].value())
            })),
        )
        .expect("the forces of an element");
        for (a, force) in computed.iter().enumerate() {
            for (i, component) in force.iter().enumerate() {
                forces[(i * 8 + a) * items + item] = component.value();
            }
        }
    }
    forces
}

/// Saint Venant-Kirchhoff has nothing but the four operations, and the operations are not fused,
/// so a device does what the host does with the same code, and the forces are the same to the bit.
#[test]
fn a_steel_model_has_the_forces_of_the_host_with_the_same_code_to_the_bit_on_a_device() {
    let coordinates = lattice(COLUMNS, LAYERS);
    let reference = NodalReferenceCoordinates::from(coordinates.clone());
    let (below, above) = (slab(COLUMNS, 0..2), slab(COLUMNS, 2..LAYERS));
    let model: Model<Blocks<Steel, Steel>, 3> = (
        Blocks(
            Steel::from((steel(), density(), below.clone(), &reference)),
            Steel::from((steel(), density(), above.clone(), &reference)),
        ),
        reference,
    )
        .into();
    let current = deformed(&coordinates);
    let Some(resident) = on_device(&model, false) else {
        return;
    };
    resident
        .compute(&current)
        .expect("the forces on the device");
    assert!(resident.failure().expect("the failures").is_none());
    let found = components(&resident.download().expect("the forces"));
    let mut blocks = Vec::new();
    model.device_blocks(&mut blocks);
    let mut expected = vec![[0.0; 3]; coordinates.len()];
    for (block, connectivity) in blocks.iter().zip([&below, &above]) {
        let forces =
            on_the_fly_element_forces(block, &steel(), connectivity, &coordinates, &current);
        gather(block, &forces, &mut expected);
    }
    let expected = expected.iter().flatten().copied().collect::<Vec<_>>();
    assert_eq!(found.len(), expected.len());
    found
        .iter()
        .zip(&expected)
        .enumerate()
        .for_each(|(k, (found, expected))| {
            assert_eq!(
                found.to_bits(),
                expected.to_bits(),
                "{found} and {expected} at {k}"
            );
        });
}

/// A model of a block of elements of one shape and one model has the forces of the host on a
/// device, with the operations fused or not.
macro_rules! shape {
    ($name:ident, $constitutive:ty, $model:expr, $points:literal, $nodes:literal) => {
        #[test]
        fn $name() {
            type Shape = Block<
                $constitutive,
                LinearElement<$points, $nodes>,
                $points,
                3,
                $nodes,
                $nodes,
                Quantity<Density>,
            >;
            let (connectivity, coordinates) = disjoint::<$points, $nodes>(7);
            let reference = NodalReferenceCoordinates::from(coordinates.clone());
            let model: Model<Shape, 3> = (
                Shape::from(($model, density(), connectivity, &reference)),
                reference,
            )
                .into();
            let current = deformed(&coordinates);
            let expected = components(
                &model
                    .nodal_forces(&current)
                    .expect("the forces on the host"),
            );
            for fused in [false, true] {
                let Some(resident) = on_device(&model, fused) else {
                    return;
                };
                let found = resident
                    .nodal_forces(&model, &current)
                    .expect("the forces on the device");
                agree(&components(&found), &expected, 1e-10);
            }
        }
    };
}

shape!(
    neo_hookean_hexahedra_are_computed_on_a_device,
    NeoHookean,
    rubber(),
    8,
    8
);
shape!(
    neo_hookean_tetrahedra_of_one_point_are_computed_on_a_device,
    NeoHookean,
    rubber(),
    1,
    4
);
shape!(
    neo_hookean_tetrahedra_of_four_points_are_computed_on_a_device,
    NeoHookean,
    rubber(),
    4,
    4
);
shape!(
    neo_hookean_wedges_are_computed_on_a_device,
    NeoHookean,
    rubber(),
    6,
    6
);
shape!(
    neo_hookean_pyramids_are_computed_on_a_device,
    NeoHookean,
    rubber(),
    8,
    5
);
shape!(
    steel_hexahedra_are_computed_on_a_device,
    SaintVenantKirchhoff,
    steel(),
    8,
    8
);
shape!(
    steel_tetrahedra_of_one_point_are_computed_on_a_device,
    SaintVenantKirchhoff,
    steel(),
    1,
    4
);
shape!(
    steel_tetrahedra_of_four_points_are_computed_on_a_device,
    SaintVenantKirchhoff,
    steel(),
    4,
    4
);
shape!(
    steel_wedges_are_computed_on_a_device,
    SaintVenantKirchhoff,
    steel(),
    6,
    6
);
shape!(
    steel_pyramids_are_computed_on_a_device,
    SaintVenantKirchhoff,
    steel(),
    8,
    5
);

/// A node moved through the elements that have it makes their Jacobians negative. The device
/// finds the first of them, and the error that comes back is the one that the host gives.
#[test]
fn an_element_that_fails_is_reported_as_the_host_reports_it() {
    let TwoBlocks {
        model, coordinates, ..
    } = two_blocks();
    let mut current = deformed(&coordinates);
    // A node of the bottom layer, which only the rubber block has, moved through the others.
    current[0] = [0.9, 0.9, 0.9].into();
    let expected = model
        .nodal_forces(&current)
        .expect_err("the host finds a failure");
    let Some(resident) = on_device(&model, true) else {
        return;
    };
    resident
        .compute(&current)
        .expect("the forces on the device");
    let (block, failure) = resident
        .failure()
        .expect("the failures")
        .expect("the device finds a failure");
    // What the device recorded is where the host finds the failure.
    assert_eq!(Some(block), expected.block.or(Some(0)));
    assert_eq!(Some(failure.item()), expected.item);
    assert_eq!(Some(failure.point()), expected.point);
    // And what comes back is the error of the host.
    let error = resident
        .nodal_forces(&model, &current)
        .expect_err("the device finds a failure");
    assert!(matches!(error, DeviceError::Element(_)));
    assert_eq!(error.to_string(), expected.to_string());
}

/// The forces do not depend on the calls before: the coordinates are given each time, and what is
/// summed starts from nothing.
#[test]
fn the_forces_of_a_call_do_not_depend_on_the_calls_before_it() {
    let TwoBlocks {
        model, coordinates, ..
    } = two_blocks();
    let shifted = coordinates
        .iter()
        .map(|&[x, y, z]| [x + 0.01, y, z])
        .collect::<Vec<_>>();
    let (first, second) = (deformed(&coordinates), deformed(&shifted));
    let Some(resident) = on_device(&model, true) else {
        return;
    };
    let forces = |current| components(&resident.nodal_forces(&model, current).expect("the forces"));
    let (a, b, again) = (forces(&first), forces(&second), forces(&first));
    assert_ne!(a, b);
    assert_eq!(
        a.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
        again.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
    );
}

/// Giving the device the coordinates and running are the two halves of a computation, and a run
/// starts from nothing each time, so running again gives the same forces.
#[test]
fn giving_the_coordinates_and_running_is_computing() {
    let TwoBlocks {
        model, coordinates, ..
    } = two_blocks();
    let current = deformed(&coordinates);
    let Some(resident) = on_device(&model, true) else {
        return;
    };
    let bits = |resident: &Resident| {
        components(&resident.download().expect("the forces"))
            .iter()
            .map(|x| x.to_bits())
            .collect::<Vec<_>>()
    };
    resident.compute(&current).expect("the forces");
    let computed = bits(&resident);
    resident.set_coordinates(&current).expect("the coordinates");
    resident.run().expect("the forces");
    assert_eq!(bits(&resident), computed);
    resident.run().expect("the forces");
    assert_eq!(bits(&resident), computed);
}

/// What the kernels need of a multiprocessor is what the driver says of them, so it is in bounds:
/// a thread has at most 255 registers, and the blocks that fit are some of the threads that a
/// multiprocessor has.
#[test]
fn the_kernels_report_what_they_need_of_a_multiprocessor() {
    let TwoBlocks { model, .. } = two_blocks();
    let Some(resident) = on_device(&model, true) else {
        return;
    };
    let resources = resident.resources().expect("the resources of the kernels");
    assert_eq!(
        resources.len(),
        4,
        "the forces and the second pass of two blocks"
    );
    for (name, resources) in &resources {
        eprintln!(
            "{name}: {resources:?}, {:.0}% occupancy",
            100.0 * resources.occupancy()
        );
        assert!((1..=255).contains(&resources.registers), "{name}");
        assert!(resources.blocks_per_multiprocessor >= 1, "{name}");
        assert!(
            resources.occupancy() > 0.0 && resources.occupancy() <= 1.0,
            "{name}"
        );
    }
}

type Steelwork = Work<Steel>;

fn steelwork() -> Steelwork {
    let coordinates = lattice(COLUMNS, LAYERS);
    let reference = NodalReferenceCoordinates::from(coordinates.clone());
    let block = Steel::from((steel(), density(), slab(COLUMNS, 0..LAYERS), &reference));
    Work::new(
        (block, reference.clone()).into(),
        reference,
        coordinates,
        COLUMNS,
    )
}

/// The first half of a step and the second, which the device does with its own kernels, are the
/// operations of the integrator of the library, in the same order, so that from the state and the
/// forces that the device has, the host makes the next state to the bit, whether or not the
/// forces are the same as the host's: each step is checked against the forces that the device
/// itself made.
#[test]
fn a_step_on_the_device_is_the_step_of_the_integrator_to_the_bit() {
    let work = steelwork();
    let Some(mut resident) = on_device(&work.model, false) else {
        return;
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
        .expect("the start");
    for step in 0..10 {
        let (x, v, a) = resident.dynamics_state().expect("the state");
        let (x, v, a) = (flat(&x), flat(&v), flat(&a));
        resident
            .steps(DT, 1, &Schedule::default())
            .expect("the step");
        let (x_next, v_next, a_next) = resident.dynamics_state().expect("the state");
        let forces = components(&resident.download().expect("the forces"));
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
}

/// The same, with the loads subtracted from the forces in their order and the coordinates that
/// are moved set to where they are held, which are the other operations of a step.
#[test]
fn a_step_with_loads_and_moved_coordinates_is_the_step_of_the_integrator_to_the_bit() {
    let work = steelwork();
    let loads = loads_on(&work);
    let forces = loads.device_forces().expect("loads that a device takes");
    let (imposed, components) = imposed_by(&work, forces, true, DT);
    let Some(mut resident) = on_device(&work.model, false) else {
        return;
    };
    let masses = Lumped.masses(&work.model);
    let per_node = masses.iter().map(|mass| mass.value()).collect::<Vec<_>>();
    let start = deformed(&work.coordinates);
    let targets = |row| targets_at(&work, &flat(&start), row, DT);
    let schedule = schedule_of(Some(&targets), &components);
    resident
        .start_dynamics(&masses, &start, &work.velocities(), &imposed, &schedule)
        .expect("the start");
    for step in 0..10 {
        let row = step + 1;
        let (x, v, a) = resident.dynamics_state().expect("the state");
        let (x, v, a) = (flat(&x), flat(&v), flat(&a));
        resident.steps(DT, 1, &schedule).expect("the step");
        let (x_next, v_next, a_next) = resident.dynamics_state().expect("the state");
        // The forces of the elements at the coordinates that the step made.
        resident.run().expect("the forces");
        let mut unbalanced = components_of(&resident);
        let mut weighted = components.iter();
        for load in &imposed.loads {
            match load {
                ImposedLoad::Constant(forces) => unbalanced
                    .iter_mut()
                    .zip(forces)
                    .for_each(|(unbalanced, force)| *unbalanced -= force),
                ImposedLoad::Weighted(weights) => {
                    let of_row = weighted.next().expect("the components of the load")(row);
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
}

/// The forces of the elements that the device has made last.
fn components_of(resident: &Resident) -> Vec<f64> {
    components(&resident.download().expect("the forces"))
}

/// The loads that a device takes subtract the forces that the host subtracts, in the order that
/// it does, to the bit, because they are what the host holds. A load that follows the body is not
/// one that a device takes. This needs no device.
#[test]
fn the_loads_that_a_device_takes_are_the_forces_that_the_host_subtracts() {
    let work = steelwork();
    let loads = loads_on(&work);
    let current = deformed(&work.coordinates);
    let forces = loads.device_forces().expect("loads that a device takes");
    assert_eq!(forces.len(), 3);
    // Starting from forces that are not nothing, so that the order of the subtractions shows.
    let start = work.model.nodal_forces(&current).expect("the forces");
    for row in [0, 1, 7, 100] {
        let mut expected = start.clone();
        loads
            .subtract_at(time_of(row, DT), &current, &mut expected)
            .expect("the host");
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
                    let of_time = components(time_of(row, DT));
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
    // None of the loads is a function of the body, and they make a list of three; but one that is
    // spoils the list.
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
    assert!(Pressure.device_forces().is_none());
    assert!((loads_on(&work), Pressure).device_forces().is_none());
    assert!((Pressure, loads_on(&work)).device_forces().is_none());
    let none = LoadAt::<NodalCoordinates<3>, NodalForcesSolid<3>>::device_forces(&());
    assert!(none.expect("no loads, no forces").is_empty());
}

/// A run on the device is a run of the integrator of the library: the same steps from the same
/// state, with the same loads and the same motion, end where it ends, to the precision of the
/// forces, which are made in another order of operations on the device. The coordinates that are
/// moved are where they are held, exactly.
fn against_the_host(
    work: &Steelwork,
    loads: &impl LoadAt<NodalCoordinates<3>, NodalForcesSolid<3>>,
    moved: bool,
) {
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
    let time = (0..=steps).map(|row| time_of(row, DT)).collect::<Vec<_>>();
    let (_, x, v, a) = work
        .model
        .integrate(
            Execution::Serial,
            Unchecked,
            Lumped,
            &VelocityVerlet::new(Time::seconds(DT)),
            &time,
            (start.clone(), velocities.clone()),
            loads,
            &conditions,
        )
        .expect("the run on the host");
    let forces = loads.device_forces().expect("loads that a device takes");
    let (imposed, components) = imposed_by(work, forces, moved, DT);
    let Some(mut resident) = on_device(&work.model, false) else {
        return;
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
        .expect("the start");
    resident.steps(DT, steps, &schedule).expect("the steps");
    let (found_x, found_v, found_a) = resident.dynamics_state().expect("the state");
    let (last_x, last_v, last_a) = (
        flat(x.iter().last().expect("coordinates")),
        flat(v.iter().last().expect("velocities")),
        flat(a.iter().last().expect("accelerations")),
    );
    close(&flat(&found_x), &last_x, "coordinates");
    close(&flat(&found_v), &last_v, "velocities");
    close(&flat(&found_a), &last_a, "accelerations");
    // The fixed coordinates are where they were, and the moved ones are where they are held, as
    // the host has them.
    let (found_x, started) = (flat(&found_x), flat(&start));
    assert!(work.fixed.iter().all(|&dof| found_x[dof] == started[dof]));
    if moved {
        for (&dof, target) in work.moved.iter().zip(targets_at(work, &started, steps, DT)) {
            assert_eq!(found_x[dof].to_bits(), target.to_bits());
            assert_eq!(last_x[dof].to_bits(), target.to_bits());
        }
    }
}

#[test]
fn steps_on_the_device_are_those_of_the_integrator_of_the_host() {
    against_the_host(&steelwork(), &(), false);
}

#[test]
fn steps_with_loads_and_moved_coordinates_are_those_of_the_integrator_of_the_host() {
    let work = steelwork();
    against_the_host(&work, &loads_on(&work), true);
}

/// The steps that are taken in one call are those taken in many, and those that go through more
/// rows of the tables than a device holds at a time are the same: nothing is kept between the
/// calls but the state on the device.
#[test]
fn steps_taken_together_are_those_taken_one_at_a_time() {
    let work = steelwork();
    let Some(mut resident) = on_device(&work.model, true) else {
        return;
    };
    let loads = loads_on(&work);
    let forces = loads.device_forces().expect("loads that a device takes");
    let (imposed, components) = imposed_by(&work, forces, true, DT);
    let masses = Lumped.masses(&work.model);
    let (start, velocities) = (deformed(&work.coordinates), work.velocities());
    let targets = |row| targets_at(&work, &flat(&start), row, DT);
    let schedule = schedule_of(Some(&targets), &components);
    let state = |resident: &Resident| {
        let (x, v, a) = resident.dynamics_state().expect("the state");
        [bits(&flat(&x)), bits(&flat(&v)), bits(&flat(&a))]
    };
    let run = |resident: &mut Resident, calls: usize, steps: usize| {
        resident
            .start_dynamics(&masses, &start, &velocities, &imposed, &schedule)
            .expect("the start");
        for _ in 0..calls {
            resident.steps(DT, steps, &schedule).expect("the steps");
        }
        state(resident)
    };
    // More steps than a table has rows, which is 1024 here, taken in one call and in fifteen.
    let together = run(&mut resident, 1, 1500);
    assert_eq!(run(&mut resident, 15, 100), together);
    assert_eq!(run(&mut resident, 1500, 1), together);
}

/// What is not a run of the model, or of what is imposed, is an error that says so.
#[test]
fn a_run_that_is_not_one_of_the_model_is_an_error() {
    let work = steelwork();
    let Some(mut resident) = on_device(&work.model, false) else {
        return;
    };
    let masses = Lumped.masses(&work.model);
    let (start, velocities) = (deformed(&work.coordinates), work.velocities());
    let loads = loads_on(&work);
    let mut start_with = |imposed: &Imposed, schedule: &Schedule<'_>| {
        resident.start_dynamics(&masses, &start, &velocities, imposed, schedule)
    };
    let invalid = |result: Result<(), GpuError>| match result {
        Err(GpuError::Invalid(message)) => message,
        other => panic!("expected an invalid run: {other:?}"),
    };
    // Coordinates that are moved, and nothing to say where.
    let (moving_imposed, components) =
        imposed_by(&work, loads.device_forces().expect("loads"), true, DT);
    let message = invalid(start_with(&moving_imposed, &schedule_of(None, &components)));
    assert!(message.contains("targets"), "{message}");
    // Components for loads, and not the loads that they are for.
    let targets = |row| targets_at(&work, &flat(&start), row, DT);
    let message = invalid(start_with(
        &moving_imposed,
        &Schedule {
            targets: Some(&targets),
            components: Vec::new(),
        },
    ));
    assert!(message.contains("components"), "{message}");
    // Targets for another number of coordinates.
    let short = |_| vec![0.0; 3];
    let message = invalid(start_with(
        &moving_imposed,
        &schedule_of(Some(&short), &components),
    ));
    assert!(message.contains("targets"), "{message}");
    // A degree of freedom that the model does not have.
    let outside = Imposed {
        fixed: vec![3 * work.coordinates.len()],
        ..Imposed::default()
    };
    let message = invalid(start_with(&outside, &Schedule::default()));
    assert!(message.contains("not in the model"), "{message}");
    // Loads of another number of nodes.
    let loads_of_another = Imposed {
        loads: vec![ImposedLoad::Constant(vec![0.0; 5])],
        ..Imposed::default()
    };
    let message = invalid(start_with(&loads_of_another, &Schedule::default()));
    assert!(message.contains("degrees of freedom"), "{message}");
    let weights_of_another = Imposed {
        loads: vec![ImposedLoad::Weighted(vec![0.0; 5])],
        ..Imposed::default()
    };
    let message = invalid(start_with(&weights_of_another, &Schedule::default()));
    assert!(message.contains("weights"), "{message}");
    // And a run that has not been started cannot be stepped.
    let Some(unstarted) = on_device(&work.model, false) else {
        return;
    };
    let message = invalid(unstarted.steps(DT, 1, &Schedule::default()));
    assert!(message.contains("started"), "{message}");
}

/// An element that fails is an error when the run starts, and in the steps.
#[test]
fn an_element_that_fails_in_a_run_of_dynamics_is_an_error() {
    let work = steelwork();
    let Some(mut resident) = on_device(&work.model, false) else {
        return;
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
    let nothing = Schedule::default();
    let error = resident
        .start_dynamics(&masses, &bad, &velocities, &imposed, &nothing)
        .expect_err("the start fails");
    assert!(matches!(error, GpuError::Failed { .. }), "{error}");
    resident
        .start_dynamics(&masses, &good, &velocities, &imposed, &nothing)
        .expect("the start");
    resident.steps(DT, 2, &nothing).expect("the steps");
    // The coordinates are put where an element fails, as a step that does that would.
    resident.set_coordinates(&bad).expect("the coordinates");
    let error = resident.steps(DT, 1, &nothing).expect_err("the step fails");
    assert!(matches!(error, GpuError::Failed { .. }), "{error}");
}

/// The components of the nodes are in the order of the nodes and then the components, which is
/// what a kernel indexes them by, and they are the memory of the nodes. This needs no device.
#[test]
fn the_components_of_the_nodes_are_node_by_node() {
    let coordinates = lattice(COLUMNS, LAYERS);
    let current = deformed(&coordinates);
    let components = current.components();
    assert_eq!(components.len(), 3 * coordinates.len());
    for (node, nodal) in current.iter().enumerate() {
        for component in 0..3 {
            assert_eq!(
                components[3 * node + component].to_bits(),
                nodal[component].value().to_bits()
            );
        }
    }
}

/// The forces and the state that are put into nodes that are kept are those that come in nodes
/// that are made, to the bit, whatever the nodes held before, and again after more steps: nothing
/// of what was in them is left.
#[test]
fn what_is_put_into_nodes_that_are_kept_is_what_comes_in_new_nodes() {
    let work = steelwork();
    let Some(mut resident) = on_device(&work.model, false) else {
        return;
    };
    let (first, second) = (
        deformed(&work.coordinates),
        deformed(
            &work
                .coordinates
                .iter()
                .map(|&[x, y, z]| [x + 0.01, y, z])
                .collect::<Vec<_>>(),
        ),
    );
    // Nodes that hold something else, which the forces must not depend on.
    let mut kept = NodalForcesSolid::<3>::zero(work.coordinates.len());
    kept.iter_mut().for_each(|force| *force = [7.0; 3].into());
    for current in [&first, &second, &first] {
        let new = resident
            .nodal_forces(&work.model, current)
            .expect("the forces");
        resident
            .nodal_forces_into(&work.model, current, &mut kept)
            .expect("the forces into nodes that are kept");
        assert_eq!(bits(&flat(&kept)), bits(&flat(&new)));
    }
    // And the state of a run.
    let masses = Lumped.masses(&work.model);
    let imposed = Imposed {
        fixed: work.fixed.clone(),
        ..Imposed::default()
    };
    resident
        .start_dynamics(
            &masses,
            &first,
            &work.velocities(),
            &imposed,
            &Schedule::default(),
        )
        .expect("the start");
    let nodes = work.coordinates.len();
    let mut kept = (
        NodalCoordinates::<3>::from(vec![[9.0; 3]; nodes]),
        NodalVelocities::<3>::from(vec![[9.0; 3]; nodes]),
        NodalAccelerations::<3>::from(vec![[9.0; 3]; nodes]),
    );
    for _ in 0..3 {
        resident
            .steps(DT, 2, &Schedule::default())
            .expect("the steps");
        let new = resident.dynamics_state().expect("the state");
        resident
            .dynamics_state_into(&mut kept.0, &mut kept.1, &mut kept.2)
            .expect("the state into nodes that are kept");
        assert_eq!(bits(&flat(&kept.0)), bits(&flat(&new.0)));
        assert_eq!(bits(&flat(&kept.1)), bits(&flat(&new.1)));
        assert_eq!(bits(&flat(&kept.2)), bits(&flat(&new.2)));
    }
}
