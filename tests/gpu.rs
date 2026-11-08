#![cfg(all(feature = "gpu", feature = "fem"))]

//! The blocks of a model as a device takes them, which need no device to check: the second pass
//! of the forces, which a device does by adding the forces of the elements to each node in the
//! order of the elements, is done here on the host from the layout that a device is given, and
//! must give the nodal forces of the model to the bit. And the forces on a device, which need
//! one.

use conspire::{
    constitutive::solid::{
        elastic::{DeviceModel, Elastic},
        hyperelastic::{NeoHookean, SaintVenantKirchhoff},
    },
    fem::{
        Blocks, Model, NodalCoordinates, NodalReferenceCoordinates,
        block::Block,
        element::{
            ElementNodalCoordinates, ElementNodalReferenceCoordinates, FiniteElement,
            FiniteElementError, OnTheFly,
            linear::{Hexahedron, LinearElement, Tetrahedron},
            solid::{
                ElementNodalForcesSolid,
                elastic::{ElasticElement, ElasticFiniteElement},
            },
        },
        solid::{NodalForcesSolid, elastic::ElasticElements},
    },
    gpu::{DeviceBlock, DeviceBlocks, DeviceError, GpuError, Resident},
    math::{Quantity, Tensor},
    units::{Density, Stress},
};

type Grid = Block<NeoHookean, Hexahedron, 8, 3, 8, 8, Quantity<Density>>;
type Steel = Block<SaintVenantKirchhoff, Hexahedron, 8, 3, 8, 8, Quantity<Density>>;
type Tetrahedra = Block<NeoHookean, Tetrahedron<1>, 1, 3, 4, 4, Quantity<Density>>;

const COLUMNS: usize = 3;
const LAYERS: usize = 4;

fn density() -> Quantity<Density> {
    Density::kilograms_per_cubic_meter(7.8e3)
}

fn rubber() -> NeoHookean {
    NeoHookean {
        shear_modulus: Stress::pascals(3.0e9),
        bulk_modulus: Stress::pascals(13.0e9),
    }
}

fn steel() -> SaintVenantKirchhoff {
    SaintVenantKirchhoff {
        shear_modulus: Stress::pascals(80.0e9),
        bulk_modulus: Stress::pascals(160.0e9),
    }
}

/// Hexahedra filling the layers `layers` of a grid of `COLUMNS` by `COLUMNS` columns.
fn slab(layers: core::ops::Range<usize>) -> Vec<[usize; 8]> {
    let index = |i: usize, j: usize, k: usize| (k * (COLUMNS + 1) + j) * (COLUMNS + 1) + i;
    layers
        .flat_map(|k| {
            (0..COLUMNS).flat_map(move |j| {
                (0..COLUMNS).map(move |i| {
                    [
                        index(i, j, k),
                        index(i + 1, j, k),
                        index(i + 1, j + 1, k),
                        index(i, j + 1, k),
                        index(i, j, k + 1),
                        index(i + 1, j, k + 1),
                        index(i + 1, j + 1, k + 1),
                        index(i, j + 1, k + 1),
                    ]
                })
            })
        })
        .collect()
}

fn lattice() -> Vec<[f64; 3]> {
    (0..=LAYERS)
        .flat_map(|k| {
            (0..=COLUMNS).flat_map(move |j| {
                (0..=COLUMNS).map(move |i| {
                    [
                        i as f64 / COLUMNS as f64,
                        j as f64 / COLUMNS as f64,
                        k as f64 / LAYERS as f64,
                    ]
                })
            })
        })
        .collect()
}

/// A smooth, small displacement so that every element has a nonzero stress.
fn deformed(coordinates: &[[f64; 3]]) -> NodalCoordinates<3> {
    coordinates
        .iter()
        .map(|&[x, y, z]| {
            [
                x + 3e-2 * (3.0 * y).sin(),
                y + 2e-2 * (2.0 * z).sin(),
                z + 4e-2 * (4.0 * x).sin(),
            ]
        })
        .collect::<Vec<_>>()
        .into()
}

/// Two blocks, and what they were made of.
struct TwoBlocks {
    model: Model<Blocks<Grid, Steel>, 3>,
    coordinates: Vec<[f64; 3]>,
    below: Vec<[usize; 8]>,
    above: Vec<[usize; 8]>,
}

/// Two blocks that share the nodes of the layer between them: rubber below and steel above.
fn two_blocks() -> TwoBlocks {
    let coordinates = lattice();
    let reference = NodalReferenceCoordinates::from(coordinates.clone());
    let (below, above) = (slab(0..2), slab(2..LAYERS));
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
    let coordinates = lattice();
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

fn skip<T>(why: &str) -> Option<T> {
    assert!(
        std::env::var_os("CONSPIRE_REQUIRE_GPU").is_none(),
        "CONSPIRE_REQUIRE_GPU is set and {why}"
    );
    eprintln!("skipped: {why}");
    None
}

/// The model on the first device, with the operations of the kernels fused or not.
fn on_device<B: DeviceBlocks>(model: &Model<B, 3>, fused: bool) -> Option<Resident> {
    match Resident::new(0, model, fused) {
        Ok(resident) => Some(resident),
        Err(GpuError::Unavailable(message)) => skip(&format!("there is no device: {message}")),
        Err(GpuError::KernelsNotBuilt) => skip("the kernels are not built"),
        Err(error) => panic!("{error}"),
    }
}

fn components(forces: &NodalForcesSolid<3>) -> Vec<f64> {
    forces
        .iter()
        .flat_map(|force| force.iter().map(|component| component.value()))
        .collect()
}

/// That the forces agree to a fraction of the largest of them.
fn agree(found: &[f64], expected: &[f64], tolerance: f64) {
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
    assert!(largest > 1e3, "the forces are {largest}, which is nothing");
    assert!(
        worst <= tolerance * largest,
        "the forces differ by {worst} of up to {largest}"
    );
}

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
    let coordinates = lattice();
    let reference = NodalReferenceCoordinates::from(coordinates.clone());
    let (below, above) = (slab(0..2), slab(2..LAYERS));
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

/// Elements that share no nodes, each the parametric element of its shape moved and bent a little
/// differently, which are valid by construction.
fn disjoint<const G: usize, const N: usize>(count: usize) -> (Vec<[usize; N]>, Vec<[f64; 3]>)
where
    LinearElement<G, N>: FiniteElement<G, 3, N, N>,
{
    let parametric = <LinearElement<G, N> as FiniteElement<G, 3, N, N>>::parametric_reference();
    let (mut connectivity, mut coordinates) = (Vec::new(), Vec::new());
    for element in 0..count {
        let mut nodes = [0; N];
        for (a, xi) in parametric.iter().enumerate() {
            let xi: [f64; 3] = core::array::from_fn(|k| xi[k].value());
            nodes[a] = coordinates.len();
            let wobble = |i: usize| 0.03 * (1.7 * a as f64 + i as f64 + element as f64).sin();
            coordinates.push([
                4.0 * element as f64 + 1.1 * xi[0] + 0.2 * xi[1] - 0.1 * xi[2] + wobble(0),
                0.05 * xi[0] + 0.9 * xi[1] + 0.15 * xi[2] + wobble(1),
                -0.1 * xi[0] + 0.12 * xi[1] + 1.2 * xi[2] + wobble(2),
            ]);
        }
        connectivity.push(nodes);
    }
    (connectivity, coordinates)
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
