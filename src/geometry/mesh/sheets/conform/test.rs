use crate::{
    geometry::{
        Coordinate,
        mesh::{
            Connectivities, Connectivity, Interfaces, Mesh, Tessellation, Verdict, Watertight,
            test::{solid, sphere},
        },
        ntree::Balancing,
        solid::Partition,
    },
    math::{FxHashSet, Tensor},
};
use core::{array::from_fn, f64::consts::PI};

const CENTERS: [[f64; 3]; 2] = [[-0.12, 0.011, 0.007], [0.16, -0.009, 0.013]];
const RADII: [f64; 2] = [0.22, 0.2];

fn ball(center: [f64; 3], tessellation: Tessellation) -> Watertight {
    let surface = tessellation.mesh();
    let points: Vec<Coordinate<3>> = surface
        .coordinates()
        .iter()
        .map(|point| from_fn(|axis| point[axis].value() + center[axis]).into())
        .collect();
    let faces: Vec<[usize; 3]> = surface
        .connectivities()
        .iter()
        .flatten()
        .map(|nodes| [nodes[0], nodes[1], nodes[2]])
        .collect();
    Watertight::new((faces, points).into()).unwrap()
}

fn box_with_one_ball() -> Vec<Watertight> {
    vec![
        solid([-0.5; 3], [0.5; 3]),
        ball([0.0; 3], sphere(24, 48, 0.25)),
    ]
}

fn box_with_two_balls() -> Vec<Watertight> {
    vec![
        solid([-0.5; 3], [0.5; 3]),
        ball(CENTERS[0], sphere(40, 80, RADII[0])),
        ball(CENTERS[1], sphere(41, 83, RADII[1])),
    ]
}

fn conformed(solids: fn() -> Vec<Watertight>, n: usize) -> (Mesh<3>, Vec<usize>) {
    let mut mesh = Mesh::from_partition(
        &Partition::new(solids()),
        &Coordinate::from([-0.5; 3]),
        &Coordinate::from([0.5; 3]),
        [n; 3],
    )
    .unwrap();
    let originals = mesh.iter().map(Connectivity::number_of_elements).collect();
    mesh.conform(&Interfaces::new(&solids()).unwrap()).unwrap();
    (mesh, originals)
}

fn dual_ball(scale: f64) -> Mesh<3> {
    let (background, _) = sphere(24, 48, 0.25)
        .dual_background(Balancing::Strong(1), scale, None)
        .unwrap();
    let coordinates = background.coordinates();
    let mut blocks = [Vec::new(), Vec::new()];
    background.iter().flatten().for_each(|element| {
        let radius = (0..3)
            .map(|axis| {
                let middle = element
                    .iter()
                    .map(|&node| coordinates[node][axis].value())
                    .sum::<f64>()
                    / 8.0;
                middle * middle
            })
            .sum::<f64>()
            .sqrt();
        blocks[usize::from(radius < 0.25)].push(from_fn(|node| element[node]))
    });
    let blocks: Vec<Connectivity> = blocks
        .into_iter()
        .map(|hexahedra: Vec<[usize; 8]>| Connectivity::Hexahedral(hexahedra.into()))
        .collect();
    (
        Connectivities::from((blocks, vec![0, 1])),
        coordinates.clone(),
    )
        .into()
}

fn conformed_dual(scale: f64) -> (Mesh<3>, Vec<usize>) {
    let mut mesh = dual_ball(scale);
    let originals = mesh.iter().map(Connectivity::number_of_elements).collect();
    mesh.conform(&Interfaces::new(&box_with_one_ball()).unwrap())
        .unwrap();
    (mesh, originals)
}

fn sheet_quality(mesh: &Mesh<3>, originals: &[usize]) -> f64 {
    let layers: FxHashSet<usize> = mesh
        .iter()
        .zip(originals)
        .flat_map(|(block, &count)| block.iter().skip(count).flatten().copied())
        .collect();
    mesh.minimum_scaled_jacobians()
        .iter()
        .zip(mesh.iter())
        .flat_map(|(qualities, block)| {
            qualities
                .iter()
                .zip(block.iter())
                .filter(|(_, element)| element.iter().any(|node| layers.contains(node)))
                .map(|(&quality, _)| quality)
                .collect::<Vec<_>>()
        })
        .fold(f64::INFINITY, f64::min)
}

fn least_quality(mesh: &Mesh<3>) -> f64 {
    mesh.minimum_scaled_jacobians()
        .iter()
        .flatten()
        .copied()
        .fold(f64::INFINITY, f64::min)
}

fn volumes(mesh: &Mesh<3>) -> Vec<f64> {
    mesh.volumes()
        .iter()
        .map(|block| block.iter().sum())
        .collect()
}

fn ball_volume(radius: f64) -> f64 {
    4.0 / 3.0 * PI * radius.powi(3)
}

#[test]
fn a_ball_in_a_box_is_conformed_without_losing_quality_or_volume() {
    let (mesh, originals) = conformed(box_with_one_ball, 12);
    assert_eq!(mesh.block_labels(), Some(&[0, 1][..]));
    assert!(least_quality(&mesh) > 0.3);
    assert!(sheet_quality(&mesh, &originals) > 0.35);
    let volumes = volumes(&mesh);
    assert!((volumes.iter().sum::<f64>() - 1.0).abs() < 1.0e-3);
    assert!((volumes[1] / ball_volume(0.25) - 1.0).abs() < 0.02);
}

#[test]
fn two_overlapping_balls_in_a_box_are_conformed_along_the_triple_line() {
    let (mesh, _) = conformed(box_with_two_balls, 24);
    assert_eq!(mesh.block_labels(), Some(&[0, 1, 2][..]));
    assert!(least_quality(&mesh) > 0.05);
    let volumes = volumes(&mesh);
    assert!((volumes.iter().sum::<f64>() - 1.0).abs() < 1.0e-3);
    assert!((volumes[2] / ball_volume(RADII[1]) - 1.0).abs() < 0.02);
}

#[test]
fn a_ball_on_a_graded_dual_mesh_is_conformed_as_well_as_on_a_uniform_one() {
    let (mesh, originals) = conformed_dual(10.0);
    assert_eq!(mesh.block_labels(), Some(&[0, 1][..]));
    assert!(sheet_quality(&mesh, &originals) > 0.35);
    assert!(least_quality(&mesh) > 0.25);
    assert!((volumes(&mesh)[1] / ball_volume(0.25) - 1.0).abs() < 0.02);
}

#[test]
fn a_material_without_a_surface_between_it_and_its_neighbor_is_refused() {
    let mut mesh = Mesh::from_partition(
        &Partition::new(box_with_one_ball()),
        &Coordinate::from([-0.5; 3]),
        &Coordinate::from([0.5; 3]),
        [12; 3],
    )
    .unwrap();
    let interfaces = Interfaces::new(&box_with_one_ball()[..1]).unwrap();
    assert!(mesh.conform(&interfaces).is_err());
}
