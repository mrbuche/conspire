use crate::geometry::mesh::{Connectivity, Mesh};

fn lattice(nel: [usize; 3], labels: impl Fn([usize; 3]) -> usize) -> Mesh<3> {
    let cells = (0..nel[2])
        .flat_map(|k| (0..nel[1]).flat_map(move |j| (0..nel[0]).map(move |i| [i, j, k])));
    Mesh::from_lattice_cells(
        cells.map(|cell| (cell, labels(cell))),
        nel,
        &[1.0; 3].into(),
        &[0.0; 3].into(),
    )
}

fn elements(mesh: &Mesh<3>) -> Vec<usize> {
    mesh.iter()
        .map(|block| match block {
            Connectivity::Hexahedral(hexahedra) => hexahedra.iter().count(),
            _ => panic!(),
        })
        .collect()
}

fn around_one_edge(labels: [usize; 4]) -> Mesh<3> {
    lattice([2, 2, 1], |[i, j, _]| labels[i + 2 * j])
}

#[test]
fn a_checkerboard_around_an_edge_is_relabeled_once() {
    let mut mesh = around_one_edge([0, 1, 1, 0]);
    let nodes = mesh.number_of_nodes();
    assert_eq!(elements(&mesh), [2, 2]);
    assert_eq!(mesh.resolve_pinches(), Ok(1));
    assert_eq!(elements(&mesh), [1, 3]);
    assert_eq!(mesh.number_of_nodes(), nodes);
    assert_eq!(mesh.block_labels(), Some(&[0, 1][..]));
    assert_eq!(mesh.resolve_pinches(), Ok(0));
}

#[test]
fn the_lowest_label_takes_the_higher_label_beside_it() {
    let mut mesh = around_one_edge([0, 1, 2, 3]);
    assert_eq!(mesh.resolve_pinches(), Ok(1));
    assert_eq!(mesh.block_labels(), Some(&[1, 2, 3][..]));
    assert_eq!(elements(&mesh), [1, 2, 1]);
    assert_eq!(mesh.resolve_pinches(), Ok(0));
}

#[test]
fn two_hexahedra_of_one_label_beside_each_other_around_an_edge_are_not_pinched() {
    let mut mesh = around_one_edge([0, 1, 2, 2]);
    assert_eq!(mesh.resolve_pinches(), Ok(0));
    assert_eq!(elements(&mesh), [1, 1, 2]);
}

#[test]
fn a_flat_interface_is_left_alone() {
    let mut mesh = lattice([4, 4, 4], |[i, ..]| i / 2);
    assert_eq!(mesh.resolve_pinches(), Ok(0));
    assert_eq!(elements(&mesh), [32, 32]);
}

#[test]
fn random_labels_are_relabeled_until_nothing_is_pinched() {
    let mut state = 12345_u64;
    let mut next = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (state >> 33) as usize % 3
    };
    let labels: Vec<usize> = (0..216).map(|_| next()).collect();
    let mut mesh = lattice([6; 3], |[i, j, k]| labels[i + 6 * (j + 6 * k)]);
    let nodes = mesh.number_of_nodes();
    assert!(mesh.resolve_pinches().unwrap() > 0);
    assert_eq!(mesh.number_of_elements(), 216);
    assert_eq!(mesh.number_of_nodes(), nodes);
    assert_eq!(mesh.resolve_pinches(), Ok(0));
}

#[test]
fn blocks_without_labels_or_hexahedra_are_refused() {
    let mut unlabeled: Mesh<3> = (
        vec![Connectivity::Hexahedral(
            vec![[0, 1, 2, 3, 4, 5, 6, 7]].into(),
        )],
        vec![[0.0; 3]; 8].into(),
    )
        .into();
    assert!(unlabeled.resolve_pinches().is_err());
    let mut tetrahedra =
        Mesh::from_lattice_tets([([0, 0, 0], 0)], [1; 3], &[1.0; 3].into(), &[0.0; 3].into());
    assert!(tetrahedra.resolve_pinches().is_err());
}
