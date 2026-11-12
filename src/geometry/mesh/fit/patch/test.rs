use super::super::Oracle;
use crate::{
    geometry::mesh::{Connectivity, Mesh, Tessellation},
    math::assert::{Assert, AssertionError},
};

fn plane(height: f64) -> Tessellation {
    (
        vec![[0, 1, 2], [0, 2, 3]],
        vec![
            [-1.0, -1.0, height],
            [1.0, -1.0, height],
            [1.0, 1.0, height],
            [-1.0, 1.0, height],
        ],
    )
        .into()
}

fn cube() -> Mesh<3> {
    (
        vec![Connectivity::Hexahedral(
            vec![[0, 1, 2, 3, 4, 5, 6, 7]].into(),
        )],
        vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
        ]
        .into(),
    )
        .into()
}

fn height_of_target(oracles: &[Oracle], owner: usize) -> f64 {
    let coordinates = vec![
        [-0.1, -0.1, 0.15],
        [0.1, -0.1, 0.15],
        [0.1, 0.1, 0.15],
        [-0.1, 0.1, 0.15],
    ]
    .into();
    let (point, _, _) = Oracle::targets(oracles, &[owner], &[vec![0, 1, 2, 3]], &coordinates, 1)
        .unwrap()
        .remove(0);
    point[2].value()
}

fn top_after_fitting_to(patch: usize) -> f64 {
    let (low, high) = (plane(0.9), plane(1.2));
    let mut mesh = cube();
    let top = vec![vec![4, 5, 6, 7]];
    let patches = if patch == 0 {
        [(&low, top), (&high, vec![])]
    } else {
        [(&low, vec![]), (&high, top)]
    };
    mesh.fit_patches(&[4, 5, 6, 7], &patches).unwrap();
    (4..8)
        .map(|node| mesh.coordinates()[node][2].value())
        .sum::<f64>()
        / 4.0
}

#[test]
fn a_face_is_attracted_to_the_target_of_its_own_patch_even_when_another_is_nearer()
-> Result<(), AssertionError> {
    let (near, far) = (plane(0.2), plane(0.0));
    let oracles = [Oracle::new(&far), Oracle::new(&near)];
    Assert::default().eq_within_tols(height_of_target(&oracles, 0), 0.0)?;
    Assert::default().eq_within_tols(height_of_target(&oracles, 1), 0.2)
}

#[test]
fn the_nearest_target_of_all_patches_is_the_wrong_one_without_matching()
-> Result<(), AssertionError> {
    let both: Tessellation = (
        vec![[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7]],
        vec![
            [-1.0, -1.0, 0.0],
            [1.0, -1.0, 0.0],
            [1.0, 1.0, 0.0],
            [-1.0, 1.0, 0.0],
            [-1.0, -1.0, 0.2],
            [1.0, -1.0, 0.2],
            [1.0, 1.0, 0.2],
            [-1.0, 1.0, 0.2],
        ],
    )
        .into();
    Assert::default().eq_within_tols(height_of_target(&[Oracle::new(&both)], 0), 0.2)
}

#[test]
fn fitting_moves_a_face_toward_the_target_of_its_own_patch() {
    assert!(top_after_fitting_to(0) < 1.0);
    assert!(top_after_fitting_to(1) > 1.0);
}
