use super::{Oracle, energy, scatter};
use crate::math::assert::perturbation;
use crate::{
    EPSILON,
    geometry::{
        Coordinates,
        mesh::{
            Connectivity, Mesh, Tessellation,
            quality::metrics::{hexahedron, tetrahedron},
            test::octahedron,
        },
    },
    math::{
        Reference, TensorRank1,
        assert::{Assert, AssertionError},
    },
    units::ReciprocalLength,
};
use core::array::from_fn;

fn gradient<const N: usize>(
    corners: &[(usize, [usize; 3]); N],
    mut coordinates: Coordinates<3>,
) -> Result<(), AssertionError> {
    let element: [usize; N] = from_fn(|i| i);
    for epsilon in [1.0, 1.0e-3] {
        let scattered = scatter(corners, &element, &coordinates, epsilon);
        for node in 0..N {
            let analytic = scattered[node].clone();
            let numerical = TensorRank1::<3, Reference, ReciprocalLength>::from(from_fn(|i| {
                coordinates[node][i] += perturbation(EPSILON);
                let above = energy(corners, &element, &coordinates, epsilon);
                coordinates[node][i] -= perturbation(2.0 * EPSILON);
                let below = energy(corners, &element, &coordinates, epsilon);
                coordinates[node][i] += perturbation(EPSILON);
                (above - below) / (2.0 * EPSILON)
            }));
            Assert::default().eq_within_fd_tol(analytic, &numerical)?;
        }
    }
    Ok(())
}

#[test]
fn gradient_matches_finite_difference() -> Result<(), AssertionError> {
    gradient(
        &hexahedron::CORNERS,
        Coordinates::from(vec![
            [0.03, -0.04, 0.01],
            [1.08, 0.05, -0.07],
            [1.02, 0.94, 0.11],
            [-0.06, 1.07, 0.02],
            [0.09, 0.01, 0.88],
            [0.94, -0.08, 1.04],
            [1.11, 1.03, 0.93],
            [0.05, 0.92, 1.09],
        ]),
    )
}

#[test]
fn tetrahedral_gradient_matches_finite_difference() -> Result<(), AssertionError> {
    gradient(
        &tetrahedron::CORNERS,
        Coordinates::from(vec![
            [0.03, -0.04, 0.01],
            [1.08, 0.05, -0.07],
            [0.02, 0.94, 0.11],
            [0.09, 0.01, 0.88],
        ]),
    )
}

fn targets(queries: &[[f64; 3]]) -> Vec<([f64; 3], [f64; 3])> {
    let tessellation = octahedron(0);
    let oracle = Oracle::new(&tessellation);
    let offsets = [
        [0.01, 0.0, 0.0],
        [-0.01, 0.0, 0.0],
        [0.0, 0.01, 0.0],
        [0.0, -0.01, 0.0],
    ];
    let coordinates = Coordinates::from(
        queries
            .iter()
            .flat_map(|query| offsets.map(|offset| from_fn(|i| query[i] + offset[i])))
            .collect::<Vec<[f64; 3]>>(),
    );
    let faces: Vec<Vec<usize>> = (0..queries.len())
        .map(|query| (4 * query..4 * query + 4).collect())
        .collect();
    oracle
        .targets(&faces, &coordinates, 1)
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

fn bent(angle: f64) -> Oracle<'static> {
    let (sine, cosine) = angle.sin_cos();
    let tessellation = Box::leak(Box::new(Tessellation::from(Mesh::from((
        vec![Connectivity::Triangular(vec![[0, 1, 2], [1, 0, 3]].into())],
        Coordinates::from(vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -cosine, sine],
        ]),
    )))));
    Oracle::new(tessellation)
}

fn corner(oracle: &Oracle, triangle: usize, corner: usize) -> [f64; 3] {
    from_fn(|i| oracle.corners[triangle][corner][i].value())
}

fn facet(oracle: &Oracle, triangle: usize) -> [f64; 3] {
    from_fn(|i| oracle.normals[triangle][i].value())
}

#[test]
fn shallow_bend_shares_one_normal_across_the_edge() {
    let oracle = bent(0.2);
    assert!(close(corner(&oracle, 0, 0), corner(&oracle, 1, 1)));
    assert!(close(corner(&oracle, 0, 1), corner(&oracle, 1, 0)));
    assert!(!close(corner(&oracle, 0, 0), facet(&oracle, 0)));
}

#[test]
fn sharp_bend_keeps_each_facet_normal_at_the_edge() {
    let oracle = bent(1.5);
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
    let queries: Vec<[f64; 3]> = (0..40)
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
    let coordinates = Coordinates::from(queries.clone());
    let faces: Vec<Vec<usize>> = (0..queries.len()).map(|query| vec![query]).collect();
    let (mut smooth, mut flat) = (0.0, 0.0);
    oracle
        .targets(&faces, &coordinates, 1)
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

/// Where the time of an energy fit goes, on a mesh as large as `SCALE` makes it.
///
/// Run alone and in release, so the numbers are not those of unoptimized code:
/// `cargo test --release -F geometry --lib profile_fit -- --ignored --nocapture`
///
/// - `SCALE` sets the size of the mesh, the number of cells growing about as its cube (5.0).
/// - `FREEDOM` is `whole` to fit every node, or `shell` for the boundary and its neighbours.
/// - The threads are those the process may run on, so `taskset -c 0-7` runs it on eight.
#[test]
#[ignore]
fn profile_fit() {
    use super::{
        Fitting, Freedom, Part,
        profile::{CALLS, NANOSECONDS, STEPS, UNKNOWNS, reset},
    };
    use crate::geometry::mesh::test::octahedron;
    use crate::geometry::{
        mesh::Dualization,
        ntree::{Balance, Balancing, CurvatureSizing, Octree, Pairing},
    };
    use core::sync::atomic::Ordering::Relaxed;
    use std::{env::var, time::Instant};
    let scale = var("SCALE")
        .ok()
        .and_then(|scale| scale.parse().ok())
        .unwrap_or(5.0);
    let freedom = match var("FREEDOM").as_deref() {
        Ok("shell") => Freedom::Shell,
        _ => Freedom::Whole,
    };
    let target = octahedron(2);
    let started = Instant::now();
    let mut octree = Octree::<u16, usize>::from_features(
        &target,
        scale,
        CurvatureSizing {
            tolerance: None,
            ..Default::default()
        },
        0,
    )
    .unwrap();
    octree
        .equilibrate(Balancing::Weak(1), Pairing::Regular)
        .unwrap();
    let mut mesh = octree.dualize();
    target.trim(&mut mesh).unwrap();
    let built = started.elapsed();
    let nodes = mesh.number_of_nodes();
    let elements = mesh.iter().flatten().count();
    reset();
    let started = Instant::now();
    mesh.inflate(&target, freedom, Fitting::Soft).unwrap();
    let total = started.elapsed().as_secs_f64() * 1.0e3;
    let threads = std::thread::available_parallelism().map_or(1, |threads| threads.get());
    let steps = STEPS.load(Relaxed);
    let unknowns = UNKNOWNS.load(Relaxed);
    let milliseconds = |part: Part| NANOSECONDS[part as usize].load(Relaxed) as f64 * 1.0e-6;
    let calls = |part: Part| CALLS[part as usize].load(Relaxed);
    println!();
    println!("mesh: {nodes} nodes, {elements} elements, built in {built:?}");
    println!(
        "fit: {freedom:?}, {threads} threads, last sweep {unknowns} unknowns, {steps} steps in all"
    );
    println!("total {total:.1} ms");
    println!(
        "{:<22}{:>8}{:>12}{:>12}{:>8}",
        "", "calls", "total ms", "ms/call", "share"
    );
    let row = |name: &str, number: u64, time: f64| {
        println!(
            "{name:<22}{number:>8}{time:>12.1}{:>12.3}{:>7.1}%",
            time / number.max(1) as f64,
            100.0 * time / total
        )
    };
    let nested = [Part::Objective, Part::Derivative, Part::Assign];
    for (name, part) in [
        ("sizes", Part::Sizes),
        ("targets (bvh)", Part::Targets),
        ("measure", Part::Measure),
        ("descent", Part::Descent),
    ] {
        row(name, calls(part), milliseconds(part));
    }
    println!("  of which");
    for (name, part) in [
        ("objective", Part::Objective),
        ("derivative", Part::Derivative),
        ("assign coordinates", Part::Assign),
    ] {
        row(name, calls(part), milliseconds(part));
    }
    let own =
        milliseconds(Part::Descent) - nested.iter().map(|&part| milliseconds(part)).sum::<f64>();
    row("L-BFGS own work", calls(Part::Descent), own);
    println!(
        "backtracks per step: {:.2}",
        calls(Part::Objective).saturating_sub(calls(Part::Descent)) as f64 / steps.max(1) as f64
    );
    println!(
        "L-BFGS vectors: about {:.1} MB ({} bytes per unknown node)",
        unknowns as f64 * 3.0 * 8.0 * 2.0 * 8.0 * 1.0e-6,
        3 * 8 * 2 * 8
    );
}
