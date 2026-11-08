#![cfg(all(feature = "gpu", feature = "fem"))]

//! The blocks of a model as a device takes them, which need no device to check: the second pass
//! of the forces, which a device does by adding the forces of the elements to each node in the
//! order of the elements, is done here on the host from the layout that a device is given, and
//! must give the nodal forces of the model to the bit. And the forces on a device, which need
//! one.

mod common;

use common::{
    agree, bits, components, deformed, density, disjoint, flat, lattice, on_device, rubber, slab,
    steel,
};
use conspire::{
    constitutive::solid::{
        elastic::{DeviceModel, Elastic},
        hyperelastic::{NeoHookean, SaintVenantKirchhoff},
    },
    fem::{
        Blocks, Model, NodalCoordinates, NodalReferenceCoordinates, NodalVelocities,
        block::Block,
        boundary::{BoundaryConditions, ConstrainAt, Displacement, Solid},
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
            ElasticExplicitDynamics, Execution, Lumped, MassChoice, Unchecked,
            elastic::ElasticElements,
        },
    },
    gpu::{DeviceBlock, DeviceBlocks, DeviceError, GpuError, Resident},
    math::{Quantity, Tensor, integrate::VelocityVerlet},
    units::{Density, Time},
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

/// A step that a time grid of these has is exact in binary, so that the steps of the integrator
/// of the host, which takes each from the difference of times, are all this.
const DT: f64 = 1.0 / 1_048_576.0;

/// A block of steel on a grid, held at its bottom layer.
struct Steelwork {
    model: Model<Steel, 3>,
    coordinates: Vec<[f64; 3]>,
    held: Vec<usize>,
}

fn steelwork() -> Steelwork {
    let coordinates = lattice(COLUMNS, LAYERS);
    let reference = NodalReferenceCoordinates::from(coordinates.clone());
    let block = Steel::from((steel(), density(), slab(COLUMNS, 0..LAYERS), &reference));
    Steelwork {
        model: (block, reference).into(),
        held: (0..3 * (COLUMNS + 1) * (COLUMNS + 1)).collect(),
        coordinates,
    }
}

/// A velocity that varies from node to node.
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

/// The first half of a step and the second, which the device does with its own kernels, are the
/// operations of the integrator of the library, in the same order, so that from the state and the
/// forces that the device has, the host makes the next state to the bit, whether or not the
/// forces are the same as the host's: each step is checked against the forces that the device
/// itself made.
#[test]
fn a_step_on_the_device_is_the_step_of_the_integrator_to_the_bit() {
    let Steelwork {
        model,
        coordinates,
        held,
    } = steelwork();
    let Some(mut resident) = on_device(&model, false) else {
        return;
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
        .expect("the start");
    for step in 0..10 {
        let (x, v, a) = resident.dynamics_state().expect("the state");
        let (x, v, a) = (flat(&x), flat(&v), flat(&a));
        resident.steps(DT, 1).expect("the step");
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
}

/// A run on the device is a run of the integrator of the library: the same steps from the same
/// state, with the held degrees of freedom held, end where it ends, to the precision of the
/// forces, which are made in another order of operations on the device.
#[test]
fn steps_on_the_device_are_those_of_the_integrator_of_the_host() {
    let Steelwork {
        model,
        coordinates,
        held,
    } = steelwork();
    let steps = 20;
    let (start, velocities) = (deformed(&coordinates), moving(&coordinates));
    let base = 0..(COLUMNS + 1) * (COLUMNS + 1);
    let conditions = BoundaryConditions::<Solid<3>>::new()
        .fix_nodes(Displacement::<0>, base.clone())
        .fix_nodes(Displacement::<1>, base.clone())
        .fix_nodes(Displacement::<2>, base);
    let time = (0..=steps)
        .map(|step| Time::seconds(step as f64 * DT))
        .collect::<Vec<_>>();
    let (_, x, v, a) = model
        .integrate(
            Execution::Serial,
            Unchecked,
            Lumped,
            &VelocityVerlet::new(Time::seconds(DT)),
            &time,
            (start.clone(), velocities.clone()),
            &(),
            &conditions,
        )
        .expect("the run on the host");
    let Some(mut resident) = on_device(&model, false) else {
        return;
    };
    resident
        .start_dynamics(&Lumped.masses(&model), &held, &start, &velocities)
        .expect("the start");
    resident.steps(DT, steps).expect("the steps");
    let (found_x, found_v, found_a) = resident.dynamics_state().expect("the state");
    let close = |found: &[f64], expected: &[f64], what: &str| {
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
        assert!(
            worst <= 1e-9 * largest,
            "the {what} differ by {worst} of up to {largest}"
        );
    };
    close(
        &flat(&found_x),
        &flat(x.iter().last().expect("coordinates")),
        "coordinates",
    );
    close(
        &flat(&found_v),
        &flat(v.iter().last().expect("velocities")),
        "velocities",
    );
    close(
        &flat(&found_a),
        &flat(a.iter().last().expect("accelerations")),
        "accelerations",
    );
    // And the held degrees of freedom are where they were.
    let moved = flat(&found_x);
    let started = flat(&start);
    assert!(held.iter().all(|&dof| moved[dof] == started[dof]));
}

/// The steps that are taken in one call are those taken in many: nothing is kept between the
/// calls but the state on the device.
#[test]
fn steps_taken_together_are_those_taken_one_at_a_time() {
    let Steelwork {
        model,
        coordinates,
        held,
    } = steelwork();
    let Some(mut resident) = on_device(&model, true) else {
        return;
    };
    let masses = Lumped.masses(&model);
    let (start, velocities) = (deformed(&coordinates), moving(&coordinates));
    let state = |resident: &Resident| {
        let (x, v, a) = resident.dynamics_state().expect("the state");
        [bits(&flat(&x)), bits(&flat(&v)), bits(&flat(&a))]
    };
    resident
        .start_dynamics(&masses, &held, &start, &velocities)
        .expect("the start");
    resident.steps(DT, 6).expect("the steps");
    let together = state(&resident);
    resident
        .start_dynamics(&masses, &held, &start, &velocities)
        .expect("the start");
    for _ in 0..3 {
        resident.steps(DT, 2).expect("the steps");
    }
    assert_eq!(state(&resident), together);
    resident
        .start_dynamics(&masses, &held, &start, &velocities)
        .expect("the start");
    for _ in 0..6 {
        resident.steps(DT, 1).expect("the step");
    }
    assert_eq!(state(&resident), together);
}

/// An element that fails is an error when the run starts, and in the steps.
#[test]
fn an_element_that_fails_in_a_run_of_dynamics_is_an_error() {
    let Steelwork {
        model,
        coordinates,
        held,
    } = steelwork();
    let Some(mut resident) = on_device(&model, false) else {
        return;
    };
    let masses = Lumped.masses(&model);
    let velocities = moving(&coordinates);
    let good = deformed(&coordinates);
    let mut bad = good.clone();
    // A node of the bottom layer moved through the elements that have it.
    bad[1] = [0.9, 0.9, 0.9].into();
    let error = resident
        .start_dynamics(&masses, &held, &bad, &velocities)
        .expect_err("the start fails");
    assert!(matches!(error, GpuError::Failed { .. }), "{error}");
    resident
        .start_dynamics(&masses, &held, &good, &velocities)
        .expect("the start");
    resident.steps(DT, 2).expect("the steps");
    // The coordinates are put where an element fails, as a step that does that would.
    resident.set_coordinates(&bad).expect("the coordinates");
    let error = resident.steps(DT, 1).expect_err("the step fails");
    assert!(matches!(error, GpuError::Failed { .. }), "{error}");
    // A run that has not been started cannot be stepped.
    let Some(unstarted) = on_device(&model, false) else {
        return;
    };
    assert!(unstarted.steps(DT, 1).is_err());
}
