use crate::geometry::{
    Coordinate,
    csg::Cuboid,
    mesh::{
        Connectivity, Interfaces, Mesh, Verdict, Watertight,
        test::{solid, sphere},
    },
    ntree::{Balancing, Pairing},
    solid::Partition,
};
use crate::math::Quantity;

fn point(entries: [f64; 3]) -> Coordinate<3> {
    entries.into()
}

fn cuboid(low: [f64; 3], high: [f64; 3]) -> Cuboid {
    Cuboid::new(point(low), point(high)).unwrap()
}

fn elements(mesh: &Mesh<3>) -> Vec<usize> {
    mesh.iter()
        .map(|block| match block {
            Connectivity::Hexahedral(hexahedra) => hexahedra.iter().count(),
            _ => panic!(),
        })
        .collect()
}

#[test]
fn touching_boxes_share_the_nodes_between_them() {
    let partition = Partition::new(vec![
        cuboid([0.0; 3], [1.0, 1.0, 1.0]),
        cuboid([1.0, 0.0, 0.0], [2.0, 1.0, 1.0]),
    ]);
    let mesh = Mesh::from_partition(
        &partition,
        &point([0.0; 3]),
        &point([2.0, 1.0, 1.0]),
        [4, 2, 2],
    )
    .unwrap();
    assert_eq!(mesh.block_labels(), Some(&[0, 1][..]));
    assert_eq!(elements(&mesh), [8, 8]);
    assert_eq!(mesh.number_of_nodes(), 5 * 3 * 3);
}

#[test]
fn a_later_volume_owns_the_space_it_overlaps() {
    let partition = Partition::new(vec![cuboid([0.0; 3], [4.0; 3]), cuboid([1.0; 3], [3.0; 3])]);
    let mesh =
        Mesh::from_partition(&partition, &point([0.0; 3]), &point([4.0; 3]), [4; 3]).unwrap();
    assert_eq!(elements(&mesh), [56, 8]);
    assert_eq!(mesh.number_of_nodes(), 5 * 5 * 5);
}

#[test]
fn cells_in_no_volume_are_dropped() {
    let partition = Partition::new(vec![cuboid([0.0; 3], [2.0; 3])]);
    let mesh =
        Mesh::from_partition(&partition, &point([0.0; 3]), &point([4.0; 3]), [4; 3]).unwrap();
    assert_eq!(elements(&mesh), [8]);
    assert_eq!(mesh.number_of_nodes(), 3 * 3 * 3);
}

#[test]
fn a_degenerate_lattice_is_refused() {
    let partition = Partition::new(vec![cuboid([0.0; 3], [1.0; 3])]);
    assert!(
        Mesh::from_partition(&partition, &point([0.0; 3]), &point([1.0; 3]), [0, 1, 1]).is_err()
    );
    assert!(
        Mesh::from_partition(
            &partition,
            &point([0.0; 3]),
            &point([1.0, 0.0, 1.0]),
            [1; 3]
        )
        .is_err()
    );
}

fn tets(mesh: &Mesh<3>) -> Vec<usize> {
    mesh.iter()
        .map(|block| match block {
            Connectivity::Tetrahedral(tets) => tets.iter().count(),
            _ => panic!(),
        })
        .collect()
}

#[test]
fn touching_boxes_share_the_nodes_between_them_as_tetrahedra() {
    let partition = Partition::new(vec![
        cuboid([0.0; 3], [1.0, 1.0, 1.0]),
        cuboid([1.0, 0.0, 0.0], [2.0, 1.0, 1.0]),
    ]);
    let mesh = Mesh::from_partition_tets(
        &partition,
        &point([0.0; 3]),
        &point([2.0, 1.0, 1.0]),
        [4, 2, 2],
    )
    .unwrap();
    assert_eq!(mesh.block_labels(), Some(&[0, 1][..]));
    assert_eq!(tets(&mesh), [48, 48]);
    assert_eq!(mesh.number_of_nodes(), 5 * 3 * 3);
}

#[test]
fn a_later_volume_owns_the_tetrahedra_it_overlaps() {
    let partition = Partition::new(vec![cuboid([0.0; 3], [4.0; 3]), cuboid([1.0; 3], [3.0; 3])]);
    let mesh =
        Mesh::from_partition_tets(&partition, &point([0.0; 3]), &point([4.0; 3]), [4; 3]).unwrap();
    assert_eq!(tets(&mesh), [6 * 56, 6 * 8]);
    assert_eq!(mesh.number_of_nodes(), 5 * 5 * 5);
}

#[test]
fn tetrahedra_in_no_volume_are_dropped() {
    let partition = Partition::new(vec![cuboid([0.0; 3], [2.0; 3])]);
    let mesh =
        Mesh::from_partition_tets(&partition, &point([0.0; 3]), &point([4.0; 3]), [4; 3]).unwrap();
    assert_eq!(tets(&mesh), [48]);
    assert_eq!(mesh.number_of_nodes(), 3 * 3 * 3);
}

#[test]
fn a_volume_splits_a_cell_between_its_tetrahedra() {
    let partition = Partition::new(vec![cuboid([0.0; 3], [1.55, 1.0, 1.0])]);
    let (lower, upper) = (point([0.0; 3]), point([2.0, 1.0, 1.0]));
    let mesh = Mesh::from_partition_tets(&partition, &lower, &upper, [2, 1, 1]).unwrap();
    assert_eq!(tets(&mesh), [10]);
    let cells = Mesh::from_partition(&partition, &lower, &upper, [2, 1, 1]).unwrap();
    assert_eq!(elements(&cells), [2]);
}

#[test]
fn a_degenerate_tetrahedral_lattice_is_refused() {
    let partition = Partition::new(vec![cuboid([0.0; 3], [1.0; 3])]);
    assert!(
        Mesh::from_partition_tets(&partition, &point([0.0; 3]), &point([1.0; 3]), [0, 1, 1])
            .is_err()
    );
    assert!(
        Mesh::from_partition_tets(
            &partition,
            &point([0.0; 3]),
            &point([1.0, 0.0, 1.0]),
            [1; 3]
        )
        .is_err()
    );
}

fn one_block(mesh: Mesh<3>) -> Mesh<3> {
    let (connectivities, coordinates) = mesh.into();
    let blocks = connectivities.into_blocks();
    assert_eq!(blocks.len(), 1);
    (blocks, coordinates).into()
}

#[test]
fn labeling_a_tetrahedral_block_by_a_partition_is_as_meshing_by_it() {
    let partition = Partition::new(vec![cuboid([0.0; 3], [4.0; 3]), cuboid([1.0; 3], [3.0; 3])]);
    let (lower, upper) = (point([0.0; 3]), point([4.0; 3]));
    let direct = Mesh::from_partition_tets(&partition, &lower, &upper, [4; 3]).unwrap();
    let whole = Mesh::from_partition_tets(
        &Partition::new(vec![cuboid([0.0; 3], [4.0; 3])]),
        &lower,
        &upper,
        [4; 3],
    )
    .unwrap();
    let labeled = one_block(whole).label_by_partition(&partition).unwrap();
    assert_eq!(labeled.block_labels(), Some(&[0, 1][..]));
    assert_eq!(tets(&labeled), tets(&direct));
    assert_eq!(labeled.number_of_nodes(), direct.number_of_nodes());
}

#[test]
fn labeling_drops_the_elements_and_nodes_no_volume_owns() {
    let whole = Mesh::from_partition(
        &Partition::new(vec![cuboid([0.0; 3], [4.0; 3])]),
        &point([0.0; 3]),
        &point([4.0; 3]),
        [4; 3],
    )
    .unwrap();
    let partition = Partition::new(vec![cuboid([0.0; 3], [2.0; 3])]);
    let labeled = one_block(whole).label_by_partition(&partition).unwrap();
    assert_eq!(elements(&labeled), [8]);
    assert_eq!(labeled.number_of_nodes(), 3 * 3 * 3);
}

#[test]
fn a_volume_owning_no_element_has_no_block() {
    let whole = Mesh::from_partition(
        &Partition::new(vec![cuboid([0.0; 3], [2.0; 3])]),
        &point([0.0; 3]),
        &point([2.0; 3]),
        [2; 3],
    )
    .unwrap();
    let partition = Partition::new(vec![cuboid([0.0; 3], [2.0; 3]), cuboid([5.0; 3], [6.0; 3])]);
    let labeled = one_block(whole).label_by_partition(&partition).unwrap();
    assert_eq!(labeled.block_labels(), Some(&[0][..]));
}

#[test]
fn labeling_is_refused_without_a_single_block_or_an_owner() {
    let two = Mesh::from_partition(
        &Partition::new(vec![cuboid([0.0; 3], [1.0; 3]), cuboid([1.0; 3], [2.0; 3])]),
        &point([0.0; 3]),
        &point([2.0; 3]),
        [2; 3],
    )
    .unwrap();
    let everywhere = Partition::new(vec![cuboid([0.0; 3], [2.0; 3])]);
    assert!(two.label_by_partition(&everywhere).is_err());
    let nowhere = Partition::new(vec![cuboid([9.0; 3], [10.0; 3])]);
    let one =
        Mesh::from_partition(&everywhere, &point([0.0; 3]), &point([2.0; 3]), [2; 3]).unwrap();
    assert!(one.label_by_partition(&nowhere).is_err());
}

fn box_with_a_ball() -> Vec<Watertight> {
    vec![
        solid([-0.5; 3], [0.5; 3]),
        Watertight::new(sphere(24, 48, 0.25)).unwrap(),
    ]
}

fn block_volumes(mesh: &Mesh<3>) -> Vec<f64> {
    mesh.volumes()
        .iter()
        .map(|block| block.iter().sum())
        .collect()
}

fn assert_a_ball_in_a_box(mesh: &Mesh<3>) {
    assert_eq!(mesh.block_labels(), Some(&[0, 1][..]));
    let ball = 4.0 / 3.0 * core::f64::consts::PI * 0.25_f64.powi(3);
    assert!((block_volumes(mesh)[1] / ball - 1.0).abs() < 0.05);
    assert!(tets(mesh)[0] > tets(mesh)[1]);
}

#[test]
fn a_uniform_tetrahedral_background_of_the_interfaces_is_labeled_into_a_box_and_a_ball() {
    let solids = box_with_a_ball();
    let interfaces = Interfaces::new(&solids).unwrap();
    let (background, _) = interfaces
        .surface()
        .lattice_tet_background(Quantity::new(0.1))
        .unwrap();
    let mesh = background
        .label_by_partition(&Partition::new(solids))
        .unwrap();
    assert_a_ball_in_a_box(&mesh);
}

#[test]
fn a_graded_tetrahedral_background_of_the_interfaces_is_labeled_into_a_box_and_a_ball() {
    let solids = box_with_a_ball();
    let interfaces = Interfaces::new(&solids).unwrap();
    let (background, _) = interfaces
        .surface()
        .octree_tet_background(Balancing::Strong(1), Pairing::Regular, 8.0, None)
        .unwrap();
    let mesh = background
        .label_by_partition(&Partition::new(solids))
        .unwrap();
    assert_a_ball_in_a_box(&mesh);
}
