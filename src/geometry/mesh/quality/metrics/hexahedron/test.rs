use crate::geometry::mesh::{Connectivity, Mesh, Verdict};

fn hex(coordinates: Vec<[f64; 3]>) -> Mesh<3> {
    let connectivities = vec![Connectivity::Hexahedral(
        vec![[0, 1, 2, 3, 4, 5, 6, 7]].into(),
    )];
    (connectivities, coordinates.into()).into()
}

const UNIT_CUBE: [[f64; 3]; 8] = [
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [1.0, 1.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
    [1.0, 0.0, 1.0],
    [1.0, 1.0, 1.0],
    [0.0, 1.0, 1.0],
];

#[test]
fn unit_cube_is_perfect() {
    let mesh = hex(UNIT_CUBE.to_vec());
    assert_eq!(mesh.minimum_jacobians(), vec![vec![1.0]]);
    assert_eq!(mesh.minimum_scaled_jacobians(), vec![vec![1.0]]);
    assert_eq!(mesh.edge_ratios(), vec![vec![1.0]]);
}

#[test]
fn scaled_is_normalized_jacobian_is_volume() {
    let mesh = hex(UNIT_CUBE.map(|point| point.map(|x| 2.0 * x)).to_vec());
    assert_eq!(mesh.minimum_jacobians(), vec![vec![8.0]]);
    assert_eq!(mesh.minimum_scaled_jacobians(), vec![vec![1.0]]);
    assert_eq!(mesh.edge_ratios(), vec![vec![1.0]]);
}

#[test]
fn stretched_hex_edge_ratio_is_longest_over_shortest() {
    let mesh = hex(UNIT_CUBE.map(|[x, y, z]| [x, y, 4.0 * z]).to_vec());
    assert_eq!(mesh.edge_ratios(), vec![vec![4.0]]);
}

#[test]
fn degenerate_at_center_though_positive_at_every_corner() {
    let mut coordinates = UNIT_CUBE.to_vec();
    coordinates[4] = [1.0, 1.0, 1.0];
    coordinates[5] = [0.0, 1.0, 1.0];
    coordinates[6] = [0.0, 0.0, 1.0];
    coordinates[7] = [1.0, 0.0, 1.0];
    let mesh = hex(coordinates);
    assert_eq!(mesh.minimum_scaled_jacobians(), vec![vec![0.0]]);
    assert_eq!(mesh.minimum_jacobians(), vec![vec![0.0]]);
}

#[test]
fn inverted_at_center_though_positive_at_every_corner() {
    let mesh = hex(vec![
        [-0.45, 0.1, -0.49],
        [0.41, 0.09, 0.1],
        [1.75, -0.46, 0.23],
        [0.76, 0.39, -0.14],
        [0.83, 0.47, 1.53],
        [1.18, 1.68, 1.01],
        [0.39, 0.74, 0.93],
        [0.13, -0.12, 1.95],
    ]);
    assert!(mesh.minimum_scaled_jacobians()[0][0] < -0.9);
    assert!(mesh.minimum_jacobians()[0][0] < 0.0);
}

#[test]
fn inverted_hex_is_negative() {
    let mut coordinates = UNIT_CUBE.to_vec();
    coordinates[4] = [0.0, 0.0, -1.0];
    let mesh = hex(coordinates);
    assert!(mesh.minimum_scaled_jacobians()[0][0] < 0.0);
    assert!(mesh.minimum_jacobians()[0][0] < 0.0);
}
