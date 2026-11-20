#![cfg(all(feature = "gpu", feature = "fem"))]

mod common;

use common::{
    agree, components, deformed, density, describe, disjoint, lattice, on_device, rubber, slab,
    steel,
};
use conspire::{
    constitutive::{
        FromParameters,
        solid::{
            elastic::Elastic,
            hyperelastic::{NeoHookean, SaintVenantKirchhoff},
        },
    },
    fem::{
        Blocks, Model, NodalCoordinates, NodalReferenceCoordinates,
        block::Block,
        element::{
            ElementNodalCoordinates, ElementNodalReferenceCoordinates, FiniteElementError,
            Recomputed,
            linear::{Hexahedron, LinearElement, Tetrahedron},
            quadratic::Hexahedron as QuadraticHexahedron,
            solid::{
                ElementNodalForcesSolid,
                elastic::{ElasticElement, ElasticFiniteElement},
            },
        },
        solid::elastic::ElasticElements,
    },
    gpu::{DeviceBlock, DeviceBlocks, DeviceError},
    math::{Quantity, Tensor, assert::AssertionError},
    units::Density,
};
use core::array::from_fn;

type Grid = Block<NeoHookean, Hexahedron, 8, 3, 8, 8, Quantity<Density>>;
type Steel = Block<SaintVenantKirchhoff, Hexahedron, 8, 3, 8, 8, Quantity<Density>>;
type Tetrahedra = Block<NeoHookean, Tetrahedron<1>, 1, 3, 4, 4, Quantity<Density>>;

const COLUMNS: usize = 3;
const LAYERS: usize = 4;

struct TwoBlocks {
    model: Model<Blocks<Grid, Steel>, 3>,
    coordinates: Vec<[f64; 3]>,
    below: Vec<[usize; 8]>,
    above: Vec<[usize; 8]>,
}

fn two_blocks() -> TwoBlocks {
    let coordinates = lattice(COLUMNS, LAYERS);
    let reference = coordinates.clone().into();
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
    let (tables_derivatives, tables_weights) = Recomputed::<3, 8, 8>::tables::<Hexahedron>();
    let derivatives = tables_derivatives
        .iter()
        .flat_map(|point| point.iter())
        .flat_map(|node| node.iter())
        .map(|derivative| derivative.value())
        .collect::<Vec<_>>();
    let weights = tables_weights.iter().copied().collect::<Vec<_>>();
    for (block, (name, parameters, connectivity)) in blocks.iter().zip([
        (NeoHookean::NAME, rubber().parameters(), &below),
        (SaintVenantKirchhoff::NAME, steel().parameters(), &above),
    ]) {
        assert_eq!(block.model, name);
        assert_eq!(block.parameters, parameters);
        assert_eq!(block.points, 8);
        assert_eq!(block.derivatives, derivatives);
        assert_eq!(block.weights, weights);
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
    let connectivity = vec![
        [0, 1, 2, 6],
        [0, 2, 3, 6],
        [0, 3, 7, 6],
        [0, 7, 4, 6],
        [0, 4, 5, 6],
        [0, 5, 1, 6],
    ];
    let model: Model<Tetrahedra, 3> = (
        (rubber(), density(), connectivity, &reference).into(),
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

#[test]
fn blocks_of_elements_of_any_order_are_described_the_same_way() {
    fn is_described<B: DeviceBlocks>() {}
    is_described::<Grid>();
    is_described::<Block<NeoHookean, QuadraticHexahedron, 27, 3, 27, 27, Quantity<Density>>>();
}

fn element_forces<M>(
    block: &DeviceBlock,
    model: &M,
    connectivity: &[[usize; 8]],
    reference: &[[f64; 3]],
    current: &NodalCoordinates<3>,
) -> Result<Vec<f64>, AssertionError>
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
        let element = Hexahedron::from(ElementNodalReferenceCoordinates::<8>::from(from_fn::<
            [f64; 3],
            8,
            _,
        >(
            |a| reference[nodes[a]],
        )));
        let computed = element.nodal_forces(
            model,
            &from_fn::<[f64; 3], 8, _>(|a| from_fn(|i| current[nodes[a]][i].value())).into(),
        )?;
        for (a, force) in computed.iter().enumerate() {
            for (i, component) in force.iter().enumerate() {
                forces[(i * 8 + a) * items + item] = component.value();
            }
        }
    }
    Ok(forces)
}

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
fn the_second_pass_from_the_layout_gives_the_nodal_forces_of_the_model_to_the_bit()
-> Result<(), AssertionError> {
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
    let first = element_forces(&blocks[0], &rubber(), &below, &reference, &current)?;
    gather(&blocks[0], &first, &mut nodal_forces);
    let second = element_forces(&blocks[1], &steel(), &above, &reference, &current)?;
    gather(&blocks[1], &second, &mut nodal_forces);
    let expected = model.nodal_forces(&current)?;
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
    Ok(())
}

#[test]
fn a_model_of_two_blocks_has_the_forces_of_the_host_on_a_device() -> Result<(), AssertionError> {
    let TwoBlocks {
        model, coordinates, ..
    } = two_blocks();
    let current = deformed(&coordinates);
    let expected = components(&model.nodal_forces(&current)?);
    for fused in [false, true] {
        let Some(resident) = on_device(&model, fused) else {
            return Ok(());
        };
        eprintln!("{}", resident.device_name());
        let found = resident.nodal_forces(&model, &current).map_err(describe)?;
        agree(&components(&found), &expected, 1e-10)?;
    }
    Ok(())
}

fn recomputed_element_forces(
    block: &DeviceBlock,
    model: &SaintVenantKirchhoff,
    connectivity: &[[usize; 8]],
    reference: &[[f64; 3]],
    current: &NodalCoordinates<3>,
) -> Result<Vec<f64>, AssertionError> {
    let (derivatives, weights) = Recomputed::<3, 8, 8>::tables::<Hexahedron>();
    let items = block.layout.items;
    let mut forces = vec![0.0; 3 * block.layout.plane()];
    for (item, nodes) in connectivity.iter().enumerate() {
        let data = Recomputed::new(
            from_fn::<[f64; 3], 8, _>(|a| reference[nodes[a]]).into(),
            &derivatives,
            &weights,
        );
        let computed = <Hexahedron as ElasticFiniteElement<SaintVenantKirchhoff, 8, 3, 8, 8>>::nodal_forces_from(
            &data,
            model,
            &from_fn::<[f64; 3], 8, _>(|a| {
                from_fn(|i| current[nodes[a]][i].value())
            }).into(),
        )?;
        for (a, force) in computed.iter().enumerate() {
            for (i, component) in force.iter().enumerate() {
                forces[(i * 8 + a) * items + item] = component.value();
            }
        }
    }
    Ok(forces)
}

#[test]
fn a_steel_model_has_the_forces_of_the_host_with_the_same_code_to_the_bit_on_a_device()
-> Result<(), AssertionError> {
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
        return Ok(());
    };
    resident.compute(&current).map_err(describe)?;
    assert!(resident.first_error().map_err(describe)?.is_none());
    let found = components(&resident.download().map_err(describe)?);
    let mut blocks = Vec::new();
    model.device_blocks(&mut blocks);
    let mut expected = vec![[0.0; 3]; coordinates.len()];
    for (block, connectivity) in blocks.iter().zip([&below, &above]) {
        let forces =
            recomputed_element_forces(block, &steel(), connectivity, &coordinates, &current)?;
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
    Ok(())
}

macro_rules! shape {
    ($name:ident, $constitutive:ty, $model:expr, $points:literal, $nodes:literal) => {
        #[test]
        fn $name() -> Result<(), AssertionError> {
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
            let expected = components(&model.nodal_forces(&current)?);
            for fused in [false, true] {
                let Some(resident) = on_device(&model, fused) else {
                    return Ok(());
                };
                let found = resident.nodal_forces(&model, &current).map_err(describe)?;
                agree(&components(&found), &expected, 1e-10)?;
            }
            Ok(())
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

#[test]
fn an_element_with_an_error_is_reported_as_the_host_reports_it() -> Result<(), AssertionError> {
    let TwoBlocks {
        model, coordinates, ..
    } = two_blocks();
    let mut current = deformed(&coordinates);
    current[0] = [0.9, 0.9, 0.9].into();
    let Err(expected) = model.nodal_forces(&current) else {
        return Err("the host finds an error".into());
    };
    let Some(resident) = on_device(&model, true) else {
        return Ok(());
    };
    resident.compute(&current).map_err(describe)?;
    let Some((block, record)) = resident.first_error().map_err(describe)? else {
        return Err("the device finds an error".into());
    };
    assert_eq!(Some(block), expected.block.or(Some(0)));
    assert_eq!(Some(record.item()), expected.item);
    assert_eq!(Some(record.point()), expected.point);
    let Err(error) = resident.nodal_forces(&model, &current) else {
        return Err("the device finds an error".into());
    };
    assert!(matches!(error, DeviceError::Element(_)));
    assert_eq!(error.to_string(), expected.to_string());
    Ok(())
}

#[test]
fn running_again_with_the_coordinates_that_the_device_has_gives_the_same_forces()
-> Result<(), AssertionError> {
    let TwoBlocks {
        model, coordinates, ..
    } = two_blocks();
    let current = deformed(&coordinates);
    let Some(resident) = on_device(&model, true) else {
        return Ok(());
    };
    resident.set_coordinates(&current).map_err(describe)?;
    let mut bits = Vec::new();
    for _ in 0..2 {
        resident.run().map_err(describe)?;
        let forces = components(&resident.download().map_err(describe)?);
        bits.push(forces.iter().map(|x| x.to_bits()).collect::<Vec<_>>());
    }
    assert_eq!(bits[0], bits[1]);
    Ok(())
}

#[test]
fn the_forces_of_a_call_do_not_depend_on_the_calls_before_it() -> Result<(), AssertionError> {
    let TwoBlocks {
        model, coordinates, ..
    } = two_blocks();
    let shifted = coordinates
        .iter()
        .map(|&[x, y, z]| [x + 0.01, y, z])
        .collect::<Vec<_>>();
    let (first, second) = (deformed(&coordinates), deformed(&shifted));
    let Some(resident) = on_device(&model, true) else {
        return Ok(());
    };
    let forces = |current| {
        resident
            .nodal_forces(&model, current)
            .map(|forces| components(&forces))
            .map_err(describe)
    };
    let (a, b, again) = (forces(&first)?, forces(&second)?, forces(&first)?);
    assert_ne!(a, b);
    assert_eq!(
        a.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
        again.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
    );
    Ok(())
}

#[test]
fn the_kernels_report_what_they_need_of_a_multiprocessor() -> Result<(), AssertionError> {
    let TwoBlocks { model, .. } = two_blocks();
    let Some(resident) = on_device(&model, true) else {
        return Ok(());
    };
    let resources = resident.resources().map_err(describe)?;
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
    Ok(())
}
