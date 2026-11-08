#![cfg(all(feature = "gpu", feature = "fem"))]

//! The blocks of a model as a device takes them, which need no device to check: the second pass
//! of the forces, which a device does by adding the forces of the elements to each node in the
//! order of the elements, is done here on the host from the layout that a device is given, and
//! must give the nodal forces of the model to the bit.

use conspire::{
    constitutive::solid::{
        elastic::{DeviceModel, Elastic},
        hyperelastic::{NeoHookean, SaintVenantKirchhoff},
    },
    fem::{
        Blocks, Model, NodalCoordinates, NodalReferenceCoordinates,
        block::Block,
        element::{
            ElementNodalCoordinates, ElementNodalReferenceCoordinates, FiniteElementError,
            OnTheFly,
            linear::{Hexahedron, Tetrahedron},
            solid::{ElementNodalForcesSolid, elastic::ElasticElement},
        },
        solid::elastic::ElasticElements,
    },
    gpu::{DeviceBlock, DeviceBlocks},
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
