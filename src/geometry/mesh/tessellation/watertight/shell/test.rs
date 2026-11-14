use crate::{
    geometry::{
        Coordinate,
        mesh::{
            Tessellation,
            test::{CONNECTIVITY, cuboid},
        },
    },
    math::Tensor,
};

fn cubes(cubes: &[([f64; 3], [f64; 3], bool)]) -> Tessellation {
    let mut faces: Vec<[usize; 3]> = Vec::new();
    let mut coordinates: Vec<Coordinate<3>> = Vec::new();
    for &(minimum, maximum, reversed) in cubes {
        let offset = coordinates.len();
        coordinates.extend(
            cuboid(minimum, maximum)
                .mesh()
                .coordinates()
                .iter()
                .cloned(),
        );
        faces.extend(CONNECTIVITY.iter().map(|face| {
            let [a, b, c] = face.map(|node| node + offset);
            if reversed { [a, c, b] } else { [a, b, c] }
        }));
    }
    (faces, coordinates).into()
}

#[test]
fn an_outward_cube_has_no_inward_shell() {
    assert!(
        cubes(&[([0.0; 3], [1.0; 3], false)])
            .inward_shells()
            .is_empty()
    );
}

#[test]
fn a_reversed_cube_is_an_inward_shell() {
    assert_eq!(
        cubes(&[([0.0; 3], [1.0; 3], true)]).inward_shells(),
        vec![0]
    );
}

#[test]
fn a_reversed_shell_is_found_even_when_the_net_volume_is_positive() {
    let tessellation = cubes(&[([0.0; 3], [2.0; 3], false), ([5.0; 3], [6.0; 3], true)]);
    assert!(tessellation.enclosed_volume().value() > 0.0);
    assert_eq!(tessellation.inward_shells(), vec![12]);
}

#[test]
fn a_cavity_wound_inward_is_not_a_defect() {
    assert!(
        cubes(&[([0.0; 3], [3.0; 3], false), ([1.0; 3], [2.0; 3], true),])
            .inward_shells()
            .is_empty()
    );
}

#[test]
fn a_cavity_wound_outward_is_a_defect() {
    assert_eq!(
        cubes(&[([0.0; 3], [3.0; 3], false), ([1.0; 3], [2.0; 3], false),]).inward_shells(),
        vec![12]
    );
}

#[test]
fn an_island_in_a_cavity_is_outward_again() {
    let island = |reversed| {
        cubes(&[
            ([0.0; 3], [5.0; 3], false),
            ([1.0; 3], [4.0; 3], true),
            ([2.0; 3], [3.0; 3], reversed),
        ])
        .inward_shells()
    };
    assert!(island(false).is_empty());
    assert_eq!(island(true), vec![24]);
}
