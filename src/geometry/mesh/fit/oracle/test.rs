use super::Oracle;
use crate::geometry::{
    Coordinates,
    mesh::{Connectivity, Mesh, Tessellation, test::octahedron},
};
use core::{array::from_fn, slice::from_ref};

fn targets(queries: &[[f64; 3]]) -> Vec<([f64; 3], [f64; 3])> {
    let tessellation = octahedron(0);
    let oracle = Oracle::new(&tessellation);
    let offsets = [
        [0.01, 0.0, 0.0],
        [-0.01, 0.0, 0.0],
        [0.0, 0.01, 0.0],
        [0.0, -0.01, 0.0],
    ];
    let coordinates = queries
        .iter()
        .flat_map(|query| offsets.map(|offset| from_fn(|i| query[i] + offset[i])))
        .collect::<Vec<_>>()
        .into();
    let faces: Vec<_> = (0..queries.len())
        .map(|query| (4 * query..4 * query + 4).collect())
        .collect();
    Oracle::targets(
        from_ref(&oracle),
        &vec![0; faces.len()],
        &faces,
        &coordinates,
        1,
    )
    .unwrap()
    .into_iter()
    .map(|(point, normal, _)| {
        (
            from_fn(|i| point[i].value()),
            from_fn(|i| normal[i].value()),
        )
    })
    .collect()
}

fn close(a: [f64; 3], b: [f64; 3]) -> bool {
    (0..3).all(|i| (a[i] - b[i]).abs() < 1.0e-12)
}

#[test]
fn equidistant_faces_average_to_a_symmetric_target() {
    let (point, normal) = targets(&[[0.0, 0.2, 0.2]])[0];
    let unit = 0.5_f64.sqrt();
    assert!(close(point, [0.0, 0.4, 0.4]), "{point:?}");
    assert!(close(normal, [0.0, unit, unit]), "{normal:?}");
}

#[test]
fn shared_edge_averages_the_face_normals() {
    let (point, normal) = targets(&[[0.0, 0.8, 0.8]])[0];
    let unit = 0.5_f64.sqrt();
    assert!(close(point, [0.0, 0.5, 0.5]), "{point:?}");
    assert!(close(normal, [0.0, unit, unit]), "{normal:?}");
}

#[test]
fn unique_nearest_face_keeps_its_normal() {
    let (point, normal) = targets(&[[0.3, 0.2, 0.2]])[0];
    let unit = 3.0_f64.sqrt().recip();
    assert!(close(point, [0.4, 0.3, 0.3]), "{point:?}");
    assert!(close(normal, [unit, unit, unit]), "{normal:?}");
}

fn bent(angle: f64) -> Tessellation {
    let (sine, cosine) = angle.sin_cos();
    Mesh::from((
        vec![Connectivity::Triangular(vec![[0, 1, 2], [1, 0, 3]].into())],
        vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -cosine, sine],
        ]
        .into(),
    ))
    .into()
}

fn corner(oracle: &Oracle, triangle: usize, corner: usize) -> [f64; 3] {
    from_fn(|i| oracle.corners[triangle][corner][i].value())
}

fn facet(oracle: &Oracle, triangle: usize) -> [f64; 3] {
    from_fn(|i| oracle.normals[triangle][i].value())
}

#[test]
fn shallow_bend_shares_one_normal_across_the_edge() {
    let tessellation = bent(0.2);
    let oracle = Oracle::new(&tessellation);
    assert!(close(corner(&oracle, 0, 0), corner(&oracle, 1, 1)));
    assert!(close(corner(&oracle, 0, 1), corner(&oracle, 1, 0)));
    assert!(!close(corner(&oracle, 0, 0), facet(&oracle, 0)));
}

#[test]
fn sharp_bend_keeps_each_facet_normal_at_the_edge() {
    let tessellation = bent(1.5);
    let oracle = Oracle::new(&tessellation);
    (0..2).for_each(|triangle| {
        (0..3).for_each(|at| {
            assert!(
                close(corner(&oracle, triangle, at), facet(&oracle, triangle)),
                "{triangle} {at}"
            )
        })
    });
}

#[test]
fn smooth_surfaces_return_a_normal_closer_to_the_true_one_than_the_facet() {
    let tessellation = octahedron(2);
    let oracle = Oracle::new(&tessellation);
    let queries: Vec<_> = (0..40)
        .map(|k| {
            let (u, v) = (0.37 * k as f64 + 0.2, 0.61 * k as f64 + 0.1);
            let radius = 0.9;
            [
                radius * u.cos() * v.sin(),
                radius * u.sin() * v.sin(),
                radius * v.cos(),
            ]
        })
        .collect();
    let coordinates = queries.clone().into();
    let faces: Vec<_> = (0..queries.len()).map(|query| vec![query]).collect();
    let (mut smooth, mut flat) = (0.0, 0.0);
    Oracle::targets(
        from_ref(&oracle),
        &vec![0; faces.len()],
        &faces,
        &coordinates,
        1,
    )
    .unwrap()
    .into_iter()
    .zip(&queries)
    .for_each(|((_, normal, _), query)| {
        let radial = {
            let norm = query.iter().map(|entry| entry * entry).sum::<f64>().sqrt();
            query.map(|entry| entry / norm)
        };
        let (_, index) = oracle
            .bvh
            .closest_point(
                &Coordinates::from(vec![*query])[0],
                oracle.coordinates,
                &oracle.elements,
            )
            .unwrap();
        let facet = &oracle.normals[index];
        smooth += (0..3).map(|i| normal[i].value() * radial[i]).sum::<f64>();
        flat += (0..3).map(|i| facet[i].value() * radial[i]).sum::<f64>();
    });
    assert!(smooth > flat, "smooth {smooth} vs flat {flat}");
}

#[test]
fn targets_are_mirror_symmetric() {
    let queries = [
        [0.0, 0.2, 0.2],
        [0.0, 0.8, 0.8],
        [0.3, 0.2, 0.2],
        [0.0, 0.0, 0.9],
    ];
    let expected = targets(&queries);
    for axis in 0..3 {
        let mirrored: Vec<[f64; 3]> = queries
            .iter()
            .map(|query| {
                let mut query = *query;
                query[axis] = -query[axis];
                query
            })
            .collect();
        for ((point, normal), (reflected_point, reflected_normal)) in
            expected.iter().zip(targets(&mirrored))
        {
            let (mut point, mut normal) = (*point, *normal);
            point[axis] = -point[axis];
            normal[axis] = -normal[axis];
            assert!(
                close(point, reflected_point),
                "{point:?} {reflected_point:?}"
            );
            assert!(
                close(normal, reflected_normal),
                "{normal:?} {reflected_normal:?}"
            );
        }
    }
}
