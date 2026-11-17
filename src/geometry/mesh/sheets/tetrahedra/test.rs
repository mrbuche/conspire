use crate::{
    geometry::mesh::{Connectivities, Connectivity, Mesh, Verdict},
    math::{
        FxHashMap, FxHashSet,
        assert::{Assert, AssertionError},
    },
};
use core::array::from_fn;

const FACES: [[usize; 3]; 4] = [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]];

fn lattice(nel: [usize; 3], mut labels: impl FnMut([usize; 3]) -> usize) -> Mesh<3> {
    let cells = (0..nel[2])
        .flat_map(|k| (0..nel[1]).flat_map(move |j| (0..nel[0]).map(move |i| [i, j, k])));
    Mesh::from_lattice_tets(
        cells.map(|cell| (cell, labels(cell))),
        nel,
        &[1.0; 3].into(),
        &[0.0; 3].into(),
    )
}

fn labeled_tets(mesh: &Mesh<3>) -> Vec<([usize; 4], usize)> {
    let labels = mesh.block_labels().expect("labeled blocks");
    mesh.iter()
        .zip(labels)
        .flat_map(|(block, &label)| {
            block
                .iter()
                .map(move |element| (from_fn(|node| element[node]), label))
        })
        .collect()
}

fn sheet_faces(tets: &[([usize; 4], usize)]) -> Vec<[usize; 3]> {
    let mut owners = FxHashMap::<[usize; 3], Vec<usize>>::default();
    for (index, (tet, _)) in tets.iter().enumerate() {
        for face in FACES {
            let mut key = face.map(|node| tet[node]);
            key.sort_unstable();
            owners.entry(key).or_default().push(index);
        }
    }
    owners
        .into_iter()
        .filter(|(_, sharing)| matches!(sharing[..], [a, b] if tets[a].1 != tets[b].1))
        .map(|(face, _)| face)
        .collect()
}

fn pinched_edges(mesh: &Mesh<3>) -> usize {
    let mut counts = FxHashMap::<[usize; 2], usize>::default();
    sheet_faces(&labeled_tets(mesh)).iter().for_each(|face| {
        [[0, 1], [1, 2], [0, 2]]
            .iter()
            .for_each(|&[a, b]| *counts.entry([face[a], face[b]]).or_default() += 1)
    });
    counts.values().filter(|&&count| count >= 4).count()
}

fn pinched_nodes(mesh: &Mesh<3>) -> usize {
    let tets = labeled_tets(mesh);
    let nodes: FxHashSet<usize> = sheet_faces(&tets).into_iter().flatten().collect();
    let mut around = FxHashMap::<usize, Vec<usize>>::default();
    for (index, (tet, _)) in tets.iter().enumerate() {
        tet.iter()
            .filter(|node| nodes.contains(node))
            .for_each(|&node| around.entry(node).or_default().push(index));
    }
    let shares_a_face = |a: usize, b: usize| {
        tets[a]
            .0
            .iter()
            .filter(|node| tets[b].0.contains(node))
            .count()
            >= 3
    };
    around
        .values()
        .map(|star| {
            let labels: FxHashSet<usize> = star.iter().map(|&tet| tets[tet].1).collect();
            labels
                .into_iter()
                .filter(|&label| {
                    let mine: Vec<usize> = star
                        .iter()
                        .copied()
                        .filter(|&tet| tets[tet].1 == label)
                        .collect();
                    let mut seen = FxHashSet::<usize>::from_iter([mine[0]]);
                    let mut pending = vec![mine[0]];
                    while let Some(tet) = pending.pop() {
                        for &other in &mine {
                            if shares_a_face(tet, other) && seen.insert(other) {
                                pending.push(other)
                            }
                        }
                    }
                    seen.len() < mine.len()
                })
                .count()
        })
        .sum()
}

fn volume(mesh: &Mesh<3>) -> f64 {
    mesh.volumes().iter().flatten().sum()
}

fn counts(mesh: &Mesh<3>) -> Vec<usize> {
    mesh.iter().map(Connectivity::number_of_elements).collect()
}

#[test]
fn a_checkerboard_around_an_edge_is_relabeled_until_the_edge_is_not_pinched()
-> Result<(), AssertionError> {
    let mut mesh = lattice([2, 2, 1], |[i, j, _]| [0, 1, 1, 0][i + 2 * j]);
    let nodes = mesh.number_of_nodes();
    let before = volume(&mesh);
    assert!(pinched_edges(&mesh) > 0);
    assert!(mesh.resolve_pinches().unwrap() > 0);
    assert_eq!(pinched_edges(&mesh), 0);
    assert_eq!(pinched_nodes(&mesh), 0);
    assert_eq!(mesh.number_of_nodes(), nodes);
    assert_eq!(mesh.number_of_elements(), 24);
    Assert::default().eq_within_tols(volume(&mesh), before)?;
    assert_eq!(mesh.resolve_pinches(), Ok(0));
    Ok(())
}

#[test]
fn two_corners_touching_at_a_node_are_relabeled_until_the_node_is_not_pinched() {
    let mut mesh = lattice([2, 2, 2], |cell| {
        usize::from(cell == [0; 3] || cell == [1; 3])
    });
    assert!(pinched_nodes(&mesh) > 0);
    assert!(mesh.resolve_pinches().unwrap() > 0);
    assert_eq!(pinched_edges(&mesh), 0);
    assert_eq!(pinched_nodes(&mesh), 0);
    assert_eq!(mesh.number_of_elements(), 48);
}

#[test]
fn a_mesh_that_is_not_pinched_is_left_as_it_is() {
    let mut mesh = lattice([4; 3], |cell| {
        usize::from(cell.iter().all(|index| (1..3).contains(index)))
    });
    let before = counts(&mesh);
    assert_eq!(mesh.resolve_pinches(), Ok(0));
    assert_eq!(counts(&mesh), before);
}

#[test]
fn random_labels_are_relabeled_until_nothing_is_pinched() {
    let mut state = 12345_u64;
    let mut mesh = lattice([5; 3], |_| {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (state >> 33) as usize % 3
    });
    assert!(pinched_edges(&mesh) + pinched_nodes(&mesh) > 0);
    assert!(mesh.resolve_pinches().unwrap() > 0);
    assert_eq!(pinched_edges(&mesh), 0);
    assert_eq!(pinched_nodes(&mesh), 0);
    assert_eq!(mesh.number_of_elements(), 6 * 5 * 5 * 5);
}

#[test]
fn blocks_without_labels_or_tetrahedra_are_refused() {
    let mut unlabeled = Mesh::<3>::from((
        vec![Connectivity::Tetrahedral(vec![[0, 1, 2, 3]].into())],
        vec![[0.0; 3]; 4].into(),
    ));
    assert!(unlabeled.resolve_pinches().is_err());
    let tets = lattice([2, 1, 1], |[i, ..]| i);
    let (blocks, coordinates) = tets.into();
    let mut blocks = Connectivities::into_blocks(blocks);
    blocks.push(Connectivity::Hexahedral(
        vec![[0, 1, 2, 3, 4, 5, 6, 7]].into(),
    ));
    let mut mixed = Mesh::<3>::from((Connectivities::from((blocks, vec![0, 1, 2])), coordinates));
    assert!(mixed.resolve_pinches().is_err());
}
