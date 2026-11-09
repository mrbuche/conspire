#![cfg(feature = "geometry")]

//! The meshes that the fit of a bone is given: the surface from an STL, an octree of it, the dual
//! of that, and the part of the dual that is inside the surface.
//!
//! This part only makes the meshes and says what they are, so that the sizes for the fits that
//! follow can be chosen. It reports, for each size of the octree, how long each step took, how
//! many hexahedra, nodes and faces of the boundary the mesh has, and how far the nodes of its
//! boundary are from the surface when the fit starts, in the sizes of their elements, which is
//! how hard the fit has to work.
//!
//! Run alone, in release mode:
//! `STL=/path/to/bone_tri.stl SCALES=5,10,20,40 cargo test --release -F geometry --test fit_gpu_bone -- bone_meshes --ignored --nocapture --test-threads=1`
//!
//! The environment sets what is run: `STL` is the surface, `bone_tri.stl` in the directory of the
//! library by default, `SCALES` are the sizes of the octree, as the cells over the thickness of
//! the surface that they should resolve, `5,10,20,40` by default, and `QUALITY` makes it also
//! find the worst scaled Jacobian of each mesh, which takes a while for the large ones.

use conspire::{
    geometry::{
        mesh::{Dualization, Tessellation, Verdict},
        ntree::{Balance, Balancing, CurvatureSizing, Octree, Pairing},
    },
    math::Tensor,
};
use std::{
    path::PathBuf,
    thread::{available_parallelism, scope},
    time::Instant,
};

fn surface() -> PathBuf {
    std::env::var_os("STL").map_or_else(
        || PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/bone_tri.stl")),
        PathBuf::from,
    )
}

fn scales() -> Vec<f64> {
    std::env::var("SCALES")
        .unwrap_or_else(|_| "5,10,20,40".to_string())
        .split(',')
        .filter_map(|scale| scale.trim().parse().ok())
        .collect()
}

/// The most memory that the process has used, in gigabytes, where the system says.
fn peak_memory() -> Option<f64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    let line = status.lines().find(|line| line.starts_with("VmHWM:"))?;
    let kilobytes: f64 = line.split_whitespace().nth(1)?.parse().ok()?;
    Some(kilobytes / 1.0e6)
}

/// The value at the fraction `fraction` of the way through the sorted values.
fn quantile(sorted: &[f64], fraction: f64) -> f64 {
    sorted[((sorted.len() - 1) as f64 * fraction).round() as usize]
}

#[test]
#[ignore]
fn bone_meshes() {
    let path = surface();
    if !path.exists() {
        eprintln!(
            "skipped: there is no surface at {}, which STL sets",
            path.display()
        );
        return;
    }
    let target = Tessellation::try_from(path.as_path()).expect("the surface is read");
    let triangles = target.mesh().number_of_elements();
    let threads = available_parallelism().map_or(1, |threads| threads.get());
    println!();
    println!(
        "surface {}: {triangles} triangles, {threads} threads",
        path.display()
    );
    println!(
        "{:>7}{:>9}{:>9}{:>8}{:>8}{:>11}{:>11}{:>9}{:>9}   {:>29}{:>9}{:>8}",
        "scale",
        "octree s",
        "balance s",
        "dual s",
        "trim s",
        "hexahedra",
        "nodes",
        "faces",
        "peak GB",
        "boundary start, in sizes: mean",
        "p90",
        "max"
    );
    for scale in scales() {
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
        .expect("the octree is made");
        let octree_seconds = started.elapsed().as_secs_f64();
        let started = Instant::now();
        octree
            .equilibrate(Balancing::Weak(1), Pairing::Regular)
            .expect("the octree is balanced");
        let balance_seconds = started.elapsed().as_secs_f64();
        let started = Instant::now();
        let mut mesh = octree.dualize();
        let dual_seconds = started.elapsed().as_secs_f64();
        drop(octree);
        let started = Instant::now();
        target.trim(&mut mesh).expect("the dual is trimmed");
        let trim_seconds = started.elapsed().as_secs_f64();
        // Where the nodes of the boundary start, from the surface, in the sizes of their elements.
        let faces = mesh.exterior_faces();
        let mut boundary: Vec<usize> = faces.iter().flatten().copied().collect();
        boundary.sort_unstable();
        boundary.dedup();
        let coordinates = mesh.coordinates();
        let neighbors = mesh.node_node_connectivity();
        let surface = target.mesh();
        let surface_coordinates = surface.coordinates();
        let elements: Vec<&[usize]> = surface.connectivities().iter().flatten().collect();
        let bvh = target.bvh();
        let chunk = boundary.len().div_ceil(threads).max(1);
        let mut ratios: Vec<f64> = scope(|scope| {
            boundary
                .chunks(chunk)
                .map(|nodes| {
                    let (coordinates, neighbors, elements) = (coordinates, neighbors, &elements);
                    scope.spawn(move || {
                        nodes
                            .iter()
                            .map(|&node| {
                                let (point, _) = bvh
                                    .closest_point(
                                        &coordinates[node],
                                        surface_coordinates,
                                        elements,
                                    )
                                    .expect("the surface has a point nearest to the node");
                                let distance = (&coordinates[node] - point).norm().value();
                                let length = neighbors[node]
                                    .iter()
                                    .map(|&other| {
                                        (&coordinates[other] - &coordinates[node]).norm().value()
                                    })
                                    .sum::<f64>()
                                    / neighbors[node].len().max(1) as f64;
                                distance / length
                            })
                            .collect::<Vec<f64>>()
                    })
                })
                .collect::<Vec<_>>()
                .into_iter()
                .flat_map(|handle| handle.join().unwrap())
                .collect()
        });
        ratios.sort_by(f64::total_cmp);
        let mean = ratios.iter().sum::<f64>() / ratios.len().max(1) as f64;
        println!(
            "{scale:>7}{octree_seconds:>9.2}{balance_seconds:>9.2}{dual_seconds:>8.2}{trim_seconds:>8.2}{:>11}{:>11}{:>9}{:>9.2}   {mean:>29.3}{:>9.3}{:>8.3}",
            mesh.number_of_elements(),
            mesh.number_of_nodes(),
            faces.len(),
            peak_memory().unwrap_or(f64::NAN),
            quantile(&ratios, 0.9),
            quantile(&ratios, 1.0),
        );
        if std::env::var_os("QUALITY").is_some() {
            let worst = mesh
                .minimum_scaled_jacobians()
                .iter()
                .flatten()
                .fold(f64::INFINITY, |worst, &quality| worst.min(quality));
            println!("        worst scaled Jacobian {worst:.4}");
        }
    }
}
