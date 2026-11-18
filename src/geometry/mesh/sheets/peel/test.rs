use super::super::test::lattice;
use crate::{
    geometry::mesh::{Connectivity, Mesh, Verdict},
    math::FxHashMap,
};
use core::array::from_fn;

const N: usize = 6;

fn nested(label: [usize; 3]) -> usize {
    label.iter().all(|index| (2..4).contains(index)).into()
}

fn triple_line(label: [usize; 3]) -> usize {
    let [i, j, k] = label;
    if (2..4).contains(&i) && (2..4).contains(&j) && (1..5).contains(&k) {
        i - 1
    } else {
        0
    }
}

fn face_counts(mesh: &Mesh<3>) -> FxHashMap<[usize; 4], usize> {
    let mut counts = FxHashMap::default();
    for block in mesh.iter() {
        for element in block.iter() {
            for face in block.element_faces(element) {
                let mut key: [usize; 4] = from_fn(|node| face[node]);
                key.sort_unstable();
                *counts.entry(key).or_default() += 1;
            }
        }
    }
    counts
}

fn volume(mesh: &Mesh<3>) -> f64 {
    mesh.volumes().iter().flatten().sum()
}

fn layers(mesh: &Mesh<3>, block: usize, count: usize) -> Vec<[usize; 8]> {
    let Connectivity::Hexahedral(hexahedra) = &mesh.connectivities()[block] else {
        panic!()
    };
    let skip = hexahedra.iter().count() - count;
    hexahedra
        .iter()
        .skip(skip)
        .map(|element| from_fn(|node| element[node]))
        .collect()
}

fn scale_about_center(mesh: &mut Mesh<3>, nodes: &[usize], factor: f64) {
    nodes.iter().for_each(|&node| {
        let point = &mesh.coordinates[node];
        let moved = from_fn(|axis| 3.0 + (point[axis].value() - 3.0) * factor);
        mesh.coordinates[node] = moved.into();
    });
}

#[test]
fn a_core_is_peeled_into_two_layers_along_its_surface_and_stays_conforming() {
    let mesh = lattice([N; 3], nested);
    let nodes = mesh.number_of_nodes();
    let before = volume(&mesh);
    let peeled = mesh.double_peel().unwrap();
    assert_eq!(peeled.sheets.len(), 1);
    assert_eq!(peeled.sheets[0].0, (1, 0));
    assert_eq!(peeled.sheets[0].1.len(), 24);
    assert_eq!(peeled.mesh.block_labels(), Some(&[0, 1][..]));
    assert_eq!(peeled.mesh.number_of_elements(), N.pow(3) + 2 * 24);
    assert_eq!(peeled.mesh.number_of_nodes(), nodes + 2 * 26);
    assert_eq!(
        peeled.free.len(),
        nodes + 2 * 26 - ((N + 1).pow(3) - (N - 1).pow(3))
    );
    let counts = face_counts(&peeled.mesh);
    assert!(counts.values().all(|&count| count <= 2));
    assert_eq!(
        counts.values().filter(|&&count| count == 1).count(),
        6 * N * N
    );
    assert!((volume(&peeled.mesh) - before).abs() < 1.0e-12);
}

#[test]
fn the_layers_open_into_positive_volumes_when_the_sheet_is_moved_out_of_the_core() {
    let mut peeled = lattice([N; 3], nested).double_peel().unwrap();
    let mut middle: Vec<usize> = peeled.sheets[0].1.iter().flatten().copied().collect();
    middle.sort_unstable();
    middle.dedup();
    let high: Vec<usize> = layers(&peeled.mesh, 0, 24)
        .iter()
        .flatten()
        .copied()
        .filter(|node| !middle.contains(node))
        .collect();
    scale_about_center(&mut peeled.mesh, &high, 1.2);
    scale_about_center(&mut peeled.mesh, &middle, 1.1);
    let volumes = peeled.mesh.volumes();
    assert!(
        volumes[1]
            .iter()
            .rev()
            .take(24)
            .all(|&volume| volume > 1.0e-6)
    );
    assert!(
        volumes[0]
            .iter()
            .rev()
            .take(24)
            .all(|&volume| volume > 1.0e-6)
    );
}

#[test]
fn three_blocks_meeting_along_a_line_stay_closed_and_conforming_after_the_peel() {
    let mesh = lattice([N; 3], triple_line);
    let before = volume(&mesh);
    let peeled = mesh.double_peel().unwrap();
    let pairs: Vec<(usize, usize)> = peeled.sheets.iter().map(|&(pair, _)| pair).collect();
    assert_eq!(pairs, [(1, 0), (2, 0), (2, 1)]);
    let faces: usize = peeled.sheets.iter().map(|(_, faces)| faces.len()).sum();
    assert_eq!(peeled.mesh.number_of_elements(), N.pow(3) + 2 * faces);
    let counts = face_counts(&peeled.mesh);
    assert!(counts.values().all(|&count| count <= 2));
    assert_eq!(
        counts.values().filter(|&&count| count == 1).count(),
        6 * N * N
    );
    assert!((volume(&peeled.mesh) - before).abs() < 1.0e-12);
}

#[test]
fn a_sheet_that_reaches_the_boundary_is_refused() {
    assert!(lattice([2, 1, 1], |[i, ..]| i).double_peel().is_err());
}

#[test]
fn blocks_without_labels_are_refused() {
    let unlabeled = Mesh::<3>::from((
        vec![Connectivity::Hexahedral(
            vec![[0, 1, 2, 3, 4, 5, 6, 7]].into(),
        )],
        vec![[0.0; 3]; 8].into(),
    ));
    assert!(unlabeled.double_peel().is_err());
}

fn tet_lattice(nel: [usize; 3], labels: impl Fn([usize; 3]) -> usize) -> Mesh<3> {
    let cells = (0..nel[2])
        .flat_map(|k| (0..nel[1]).flat_map(move |j| (0..nel[0]).map(move |i| [i, j, k])));
    Mesh::from_lattice_tets(
        cells.map(|cell| (cell, labels(cell))),
        nel,
        &[1.0; 3].into(),
        &[0.0; 3].into(),
    )
}

fn tet_face_counts(mesh: &Mesh<3>) -> FxHashMap<[usize; 3], usize> {
    let mut counts = FxHashMap::default();
    for block in mesh.iter() {
        for element in block.iter() {
            for face in block.element_faces(element) {
                let mut key: [usize; 3] = from_fn(|node| face[node]);
                key.sort_unstable();
                *counts.entry(key).or_default() += 1;
            }
        }
    }
    counts
}

fn tet_layers(mesh: &Mesh<3>, block: usize, count: usize) -> Vec<[usize; 4]> {
    let Connectivity::Tetrahedral(tets) = &mesh.connectivities()[block] else {
        panic!()
    };
    let skip = tets.iter().count() - count;
    tets.iter()
        .skip(skip)
        .map(|element| from_fn(|node| element[node]))
        .collect()
}

#[test]
fn a_tetrahedral_core_is_peeled_into_two_layers_along_its_surface_and_stays_conforming() {
    let mesh = tet_lattice([N; 3], nested);
    let nodes = mesh.number_of_nodes();
    let before = volume(&mesh);
    let peeled = mesh.double_peel().unwrap();
    let triangles = 2 * 24;
    assert_eq!(peeled.sheets.len(), 1);
    assert_eq!(peeled.sheets[0].0, (1, 0));
    assert_eq!(peeled.sheets[0].1.len(), triangles);
    assert_eq!(peeled.mesh.block_labels(), Some(&[0, 1][..]));
    assert_eq!(
        peeled.mesh.number_of_elements(),
        6 * N.pow(3) + 2 * 3 * triangles
    );
    assert_eq!(peeled.mesh.number_of_nodes(), nodes + 2 * 26);
    assert_eq!(
        peeled.free.len(),
        nodes + 2 * 26 - ((N + 1).pow(3) - (N - 1).pow(3))
    );
    let counts = tet_face_counts(&peeled.mesh);
    assert!(counts.values().all(|&count| count <= 2));
    assert_eq!(
        counts.values().filter(|&&count| count == 1).count(),
        12 * N * N
    );
    assert!((volume(&peeled.mesh) - before).abs() < 1.0e-12);
}

#[test]
fn the_tetrahedral_layers_open_into_positive_volumes_when_the_sheet_is_moved_out_of_the_core() {
    let mut peeled = tet_lattice([N; 3], nested).double_peel().unwrap();
    let layer = 3 * peeled.sheets[0].1.len();
    let mut middle: Vec<usize> = peeled.sheets[0].1.iter().flatten().copied().collect();
    middle.sort_unstable();
    middle.dedup();
    let high: Vec<usize> = tet_layers(&peeled.mesh, 0, layer)
        .iter()
        .flatten()
        .copied()
        .filter(|node| !middle.contains(node))
        .collect();
    scale_about_center(&mut peeled.mesh, &high, 1.2);
    scale_about_center(&mut peeled.mesh, &middle, 1.1);
    let volumes = peeled.mesh.volumes();
    [0, 1].into_iter().for_each(|block| {
        assert!(
            volumes[block]
                .iter()
                .rev()
                .take(layer)
                .all(|&volume| volume > 1.0e-6),
            "block {block}"
        )
    });
}

#[test]
fn three_tetrahedral_blocks_meeting_along_a_line_stay_closed_and_conforming_after_the_peel() {
    let mesh = tet_lattice([N; 3], triple_line);
    let before = volume(&mesh);
    let peeled = mesh.double_peel().unwrap();
    let pairs: Vec<(usize, usize)> = peeled.sheets.iter().map(|&(pair, _)| pair).collect();
    assert_eq!(pairs, [(1, 0), (2, 0), (2, 1)]);
    let faces: usize = peeled.sheets.iter().map(|(_, faces)| faces.len()).sum();
    assert_eq!(peeled.mesh.number_of_elements(), 6 * N.pow(3) + 6 * faces);
    let counts = tet_face_counts(&peeled.mesh);
    assert!(counts.values().all(|&count| count <= 2));
    assert_eq!(
        counts.values().filter(|&&count| count == 1).count(),
        12 * N * N
    );
    assert!((volume(&peeled.mesh) - before).abs() < 1.0e-12);
}

#[test]
fn a_tetrahedral_sheet_that_reaches_the_boundary_is_refused() {
    assert!(tet_lattice([2, 1, 1], |[i, ..]| i).double_peel().is_err());
}

#[test]
fn blocks_of_different_kinds_are_refused() {
    let tets = tet_lattice([N; 3], nested);
    let hexes = lattice([N; 3], nested);
    let (tet_blocks, coordinates): (_, crate::geometry::Coordinates<3>) = tets.into();
    let (hex_blocks, _) = hexes.into();
    let mut blocks = tet_blocks.into_blocks();
    blocks.truncate(1);
    blocks.push(hex_blocks.into_blocks().remove(1));
    let mixed: Mesh<3> = (
        crate::geometry::mesh::Connectivities::from((blocks, vec![0, 1])),
        coordinates,
    )
        .into();
    assert!(mixed.double_peel().is_err());
}
