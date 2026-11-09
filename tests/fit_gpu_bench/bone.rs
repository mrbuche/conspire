//! The fit of a bone: the mesh that a surface from an STL gives, as the problem that the kernels
//! take, the targets of its faces from the surface, and the fit on the threads of the host.
//!
//! The surface is made into an octree, which is dualized and trimmed to the part inside the
//! surface, as the library does before it buffers and fits. Its faces of the boundary then start
//! about a size of an element from the surface, which is how far a real fit has to take them.
//! The targets of those faces are found each sweep from where their nodes are: the nearest point
//! of the surface to the centroid of the face, and the plane of the triangle that it is on.
//!
//! `bone_fit_cpu` fits the meshes on the threads of the host, with the same method and the same
//! kernel arithmetic as the device, and `bone_fit_gpu` in the device part of this file fits them
//! on a device. Run alone, in release mode:
//! `STL=/path/to/bone_tri.stl SCALES=40,80 cargo test --release -F geometry --test fit_gpu_bench -- bone_fit_cpu --ignored --nocapture --test-threads=1`
//! `STL=/path/to/bone_tri.stl SCALES=40,80,160 CONSPIRE_REQUIRE_GPU=1 cargo test --release -F "gpu geometry" --test fit_gpu_bench -- bone_fit_gpu --ignored --nocapture --test-threads=1`
//!
//! The environment sets what is run: `STL` is the surface, `bone_tri.stl` in the directory of the
//! library by default, `SCALES` are the sizes of the octree, as the cells over the thickness of the
//! surface that they should resolve, `40,80` by default, `STEPS` is the most steps of a sweep, 100
//! by default, `SWEEPS` the most sweeps, 50 by default, `THREADS` the threads of the host, and
//! `VERBOSE` says what each sweep did.

use super::{
    Layout, Point, cross, dot, minus, random,
    solver::{Backend, Emulated, Settings, SweepBackend, Sweeps, Targets, Threaded, fit},
    surface::{Surface, surface_targets_on_host},
};
use conspire::geometry::{
    Coordinate,
    mesh::{Dualization, ExodusFormat, Input, Mesh, Output, Tessellation, Verdict},
    ntree::{Balance, Balancing, CurvatureSizing, Octree, Pairing},
};
use conspire::io::Write;
use std::{
    path::PathBuf,
    thread::{available_parallelism, scope},
    time::Instant,
};

/// Where the surface is.
pub fn surface() -> PathBuf {
    std::env::var_os("STL").map_or_else(
        || PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/bone_tri.stl")),
        PathBuf::from,
    )
}

/// The surface, or why there is none.
pub fn read() -> Option<Tessellation> {
    let path = surface();
    if !path.exists() {
        assert!(
            std::env::var_os("CONSPIRE_REQUIRE_GPU").is_none(),
            "CONSPIRE_REQUIRE_GPU is set and there is no surface at {}, which STL sets",
            path.display()
        );
        eprintln!(
            "skipped: there is no surface at {}, which STL sets",
            path.display()
        );
        return None;
    }
    Some(Tessellation::try_from(path.as_path()).expect("the surface is read"))
}

pub fn scales() -> Vec<f64> {
    std::env::var("SCALES")
        .unwrap_or_else(|_| "40,80".to_string())
        .split(',')
        .filter_map(|scale| scale.trim().parse().ok())
        .collect()
}

/// A number that the environment sets, or what it is without.
pub fn setting(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

/// The part of the dual of an octree of the surface that is inside it.
pub fn mesh(target: &Tessellation, scale: f64) -> Mesh<3> {
    let mut octree = Octree::<u16, usize>::from_features(
        target,
        scale,
        CurvatureSizing {
            tolerance: None,
            ..Default::default()
        },
        0,
    )
    .expect("the octree is made");
    octree
        .equilibrate(Balancing::Weak(1), Pairing::Regular)
        .expect("the octree is balanced");
    let mut mesh = octree.dualize();
    drop(octree);
    target.trim(&mut mesh).expect("the dual is trimmed");
    mesh
}

/// The interior mesh of a case, read from `INTERIOR` by the ending of its name: `.inp` is Abaqus,
/// `.e`, `.exo` or `.g` Exodus, `.mesh` Medit, `.vtk` or `.vtu` VTK.
fn interior(path: &std::path::Path) -> Mesh<3> {
    let extension = path
        .extension()
        .and_then(|extension| extension.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase();
    let input = match extension.as_str() {
        "inp" => Input::Abaqus(path),
        "e" | "exo" | "g" | "exodus" => Input::Exodus(path),
        "mesh" => Input::Medit(path),
        "vtk" | "vtu" => Input::VtkUnstructured(path),
        other => panic!("INTERIOR ends in `{other}`, which is not a mesh format that is read"),
    };
    Mesh::try_from(input).unwrap_or_else(|why| panic!("{} is not read: {why}", path.display()))
}

/// A case: the surface from `STL` and the mesh of the inside of it in `INTERIOR`, with the buffer
/// layer added as the library does before it fits, and not fitted. None unless `INTERIOR` is set.
pub fn case() -> Option<(Tessellation, Bone, Mesh<3>)> {
    let path = std::env::var_os("INTERIOR")?;
    let target = read()?;
    let started = Instant::now();
    let interior = interior(std::path::Path::new(&path));
    let read = started.elapsed().as_secs_f64();
    let hexahedra = interior.number_of_elements();
    let started = Instant::now();
    let (mesh, _) = interior.buffer_layer().expect("the buffer layer is added");
    let layered = started.elapsed().as_secs_f64();
    let started = Instant::now();
    let bone = layout_of(&mesh);
    let converted = started.elapsed().as_secs_f64();
    eprintln!(
        "case: {hexahedra} hexahedra in the interior, {} with the buffer layer, {} nodes, {} faces; read {read:.1} s, layer {layered:.1} s, layout {converted:.1} s",
        bone.layout.elements, bone.layout.nodes, bone.layout.faces
    );
    Some((target, bone, mesh))
}

/// What a fit of a case came to: the mesh at `x`, its worst scaled Jacobian and how far the
/// boundary is from the surface, and the mesh in the file `OUTPUT` as Exodus if that is set.
pub fn finish(mesh: Mesh<3>, nodes: usize, surface: &SurfaceTargets, x: &[f64]) {
    let started = Instant::now();
    let mut used = vec![false; mesh.number_of_nodes()];
    for element in mesh.iter().flatten() {
        for &node in element {
            used[node] = true;
        }
    }
    let (connectivities, mut coordinates) = mesh.into();
    let mut next = 0;
    for (old, used) in used.into_iter().enumerate() {
        if used {
            coordinates[old] =
                Coordinate::<3>::from([x[next], x[nodes + next], x[2 * nodes + next]]);
            next += 1;
        }
    }
    assert_eq!(next, nodes, "the nodes of the layout are those of the mesh");
    let fitted = Mesh::from((connectivities, coordinates));
    let jacobians: Vec<f64> = fitted
        .minimum_scaled_jacobians()
        .into_iter()
        .flatten()
        .collect();
    let worst = jacobians.iter().copied().fold(f64::INFINITY, f64::min);
    let inverted = jacobians
        .iter()
        .filter(|&&jacobian| jacobian <= 0.0)
        .count();
    let poor = jacobians.iter().filter(|&&jacobian| jacobian < 0.2).count();
    let mean = jacobians.iter().sum::<f64>() / jacobians.len() as f64;
    // The planes of the boundary faces, from the nodes of the fit, which says how far from the
    // surface the faces are.
    let (_, _, squares) = surface.find(x);
    let mut distances: Vec<f64> = squares.iter().map(|square| square.sqrt()).collect();
    distances.sort_by(f64::total_cmp);
    let at = |fraction: f64| distances[((distances.len() - 1) as f64 * fraction) as usize];
    eprintln!(
        "fitted mesh: scaled Jacobian least {worst:.4}, mean {mean:.4}, {inverted} not positive, {poor} below 0.2 of {}",
        jacobians.len()
    );
    eprintln!(
        "             boundary nodes off the plane of the nearest triangle: median {:.3e}, 99% {:.3e}, greatest {:.3e}",
        at(0.5),
        at(0.99),
        at(1.0)
    );
    if let Some(path) = std::env::var_os("OUTPUT") {
        fitted
            .write(Output::Exodus(ExodusFormat::Classic(path.clone())))
            .expect("the fitted mesh is written");
        eprintln!("             written to {}", path.to_string_lossy());
    }
    eprintln!(
        "             checked in {:.1} s",
        started.elapsed().as_secs_f64()
    );
}

/// A mesh as the problem that the kernels take, and where its nodes start, on planes.
pub struct Bone {
    pub layout: Layout,
    pub coordinates: Vec<f64>,
}

/// The problem of a mesh of hexahedra. Nodes that no element has are left out, since they would
/// have no size.
pub fn layout_of(mesh: &Mesh<3>) -> Bone {
    let original = mesh.number_of_nodes();
    let hexahedra: Vec<[usize; 8]> = mesh
        .iter()
        .flatten()
        .map(|element| <[usize; 8]>::try_from(element).expect("every element is a hexahedron"))
        .collect();
    let elements = hexahedra.len();
    let mut renumbered = vec![u32::MAX; original];
    hexahedra
        .iter()
        .flatten()
        .for_each(|&node| renumbered[node] = 0);
    let mut nodes = 0_u32;
    for entry in renumbered.iter_mut().filter(|entry| **entry == 0) {
        *entry = nodes;
        nodes += 1;
    }
    let nodes = nodes as usize;
    let source = mesh.coordinates();
    let mut coordinates = vec![0.0; 3 * nodes];
    for (old, &new) in renumbered.iter().enumerate() {
        if new != u32::MAX {
            for d in 0..3 {
                coordinates[d * nodes + new as usize] = source[old][d].value();
            }
        }
    }
    let mut connectivity = vec![0_u32; 8 * elements];
    for (element, nodes) in hexahedra.iter().enumerate() {
        for (corner, &node) in nodes.iter().enumerate() {
            connectivity[corner * elements + element] = renumbered[node];
        }
    }
    drop(hexahedra);
    // The corners on each node, as positions among the corners, in the order of the elements.
    let mut starts = vec![0_u32; nodes + 1];
    connectivity
        .iter()
        .for_each(|&node| starts[node as usize + 1] += 1);
    (0..nodes).for_each(|node| starts[node + 1] += starts[node]);
    let mut next = starts.clone();
    let mut entries = vec![0_u32; 8 * elements];
    for element in 0..elements {
        for corner in 0..8 {
            let node = connectivity[corner * elements + element] as usize;
            entries[next[node] as usize] = (corner * elements + element) as u32;
            next[node] += 1;
        }
    }
    // The faces of the boundary and the faces on each node.
    let boundary = mesh.exterior_faces();
    let faces = boundary.len();
    let mut face_nodes = vec![0_u32; 4 * faces];
    for (face, nodes) in boundary.iter().enumerate() {
        assert_eq!(nodes.len(), 4, "every face is a quadrilateral");
        for (corner, &node) in nodes.iter().enumerate() {
            face_nodes[corner * faces + face] = renumbered[node];
        }
    }
    drop(boundary);
    let mut face_starts = vec![0_u32; nodes + 1];
    face_nodes
        .iter()
        .for_each(|&node| face_starts[node as usize + 1] += 1);
    (0..nodes).for_each(|node| face_starts[node + 1] += face_starts[node]);
    let mut next = face_starts.clone();
    let mut face_entries = vec![0_u32; 4 * faces];
    for face in 0..faces {
        for corner in 0..4 {
            let node = face_nodes[corner * faces + face] as usize;
            face_entries[next[node] as usize] = face as u32;
            next[node] += 1;
        }
    }
    // The nodes that an edge joins to each node, each once, in order.
    const EDGES: [[usize; 2]; 12] = [
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 0],
        [4, 5],
        [5, 6],
        [6, 7],
        [7, 4],
        [0, 4],
        [1, 5],
        [2, 6],
        [3, 7],
    ];
    let mut raw = vec![0_u32; nodes + 1];
    for element in 0..elements {
        for [a, b] in EDGES {
            raw[connectivity[a * elements + element] as usize + 1] += 1;
            raw[connectivity[b * elements + element] as usize + 1] += 1;
        }
    }
    (0..nodes).for_each(|node| raw[node + 1] += raw[node]);
    let mut next = raw.clone();
    let mut joined = vec![0_u32; raw[nodes] as usize];
    for element in 0..elements {
        for [a, b] in EDGES {
            let (a, b) = (
                connectivity[a * elements + element],
                connectivity[b * elements + element],
            );
            joined[next[a as usize] as usize] = b;
            next[a as usize] += 1;
            joined[next[b as usize] as usize] = a;
            next[b as usize] += 1;
        }
    }
    let mut neighbor_starts = Vec::with_capacity(nodes + 1);
    let mut neighbor_entries = Vec::with_capacity(joined.len() / 3);
    neighbor_starts.push(0_u32);
    for node in 0..nodes {
        let row = &mut joined[raw[node] as usize..raw[node + 1] as usize];
        row.sort_unstable();
        let mut last = None;
        for &other in row.iter() {
            if last != Some(other) {
                neighbor_entries.push(other);
                last = Some(other);
            }
        }
        neighbor_starts.push(neighbor_entries.len() as u32);
    }
    drop(joined);
    Bone {
        coordinates,
        layout: Layout {
            nodes,
            elements,
            faces,
            epsilon: 1.0,
            connectivity,
            scales: vec![0.0; elements],
            lengths: vec![0.0; nodes],
            starts,
            entries,
            face_starts,
            face_entries,
            neighbor_starts,
            neighbor_entries,
            face_nodes,
            points: vec![0.0; 3 * faces],
            normals: vec![0.0; 3 * faces],
            distances: vec![0.0; faces],
        },
    }
}

/// The triangles of a surface, as the corners of each.
pub fn soup_of(target: &Tessellation) -> Vec<[Point; 3]> {
    let coordinates = target.mesh().coordinates();
    target
        .mesh()
        .connectivities()
        .iter()
        .flatten()
        .map(|triangle| {
            [0, 1, 2].map(|index| {
                let point = &coordinates[triangle[index]];
                [point[0].value(), point[1].value(), point[2].value()]
            })
        })
        .collect()
}

/// The targets of the faces of a layout, from a surface: for each face, the point of the surface
/// nearest to its centroid, the unit normal of the triangle that the point is on, and the largest
/// of the squared deviations of the nodes of the face from that plane.
pub struct SurfaceTargets<'a> {
    target: &'a Tessellation,
    /// The unit normal of each triangle.
    normals: Vec<[f64; 3]>,
    face_nodes: Vec<u32>,
    nodes: usize,
    faces: usize,
}

impl<'a> SurfaceTargets<'a> {
    pub fn new(target: &'a Tessellation, layout: &Layout) -> Self {
        let coordinates = target.mesh().coordinates();
        let normals = target
            .mesh()
            .connectivities()
            .iter()
            .flatten()
            .map(|triangle| {
                let corner = |index: usize| {
                    let point = &coordinates[triangle[index]];
                    [point[0].value(), point[1].value(), point[2].value()]
                };
                let normal = cross(minus(corner(1), corner(0)), minus(corner(2), corner(0)));
                let length = dot(normal, normal).sqrt();
                if length > 0.0 {
                    normal.map(|c| c / length)
                } else {
                    [0.0; 3]
                }
            })
            .collect();
        Self {
            target,
            normals,
            face_nodes: layout.face_nodes.clone(),
            nodes: layout.nodes,
            faces: layout.faces,
        }
    }
    /// The points, the normals and the squared distances, on planes, for the nodes at `x`.
    pub fn find(&self, x: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let (nodes, faces) = (self.nodes, self.faces);
        let surface = self.target.mesh();
        let surface_coordinates = surface.coordinates();
        let elements: Vec<&[usize]> = surface.connectivities().iter().flatten().collect();
        let bvh = self.target.bvh();
        let (triangle_normals, face_nodes) = (&self.normals, &self.face_nodes);
        let threads = available_parallelism().map_or(1, |threads| threads.get());
        let mut records = vec![[0.0_f64; 7]; faces];
        let chunk = faces.div_ceil(threads).max(1);
        scope(|scope| {
            for (run, records) in records.chunks_mut(chunk).enumerate() {
                let elements = &elements;
                scope.spawn(move || {
                    for (offset, record) in records.iter_mut().enumerate() {
                        let face = run * chunk + offset;
                        let mut centroid = [0.0; 3];
                        for corner in 0..4 {
                            let node = face_nodes[corner * faces + face] as usize;
                            for d in 0..3 {
                                centroid[d] += x[d * nodes + node] * 0.25;
                            }
                        }
                        let (point, triangle) = bvh
                            .closest_point(
                                &Coordinate::<3>::from(centroid),
                                surface_coordinates,
                                elements,
                            )
                            .expect("the surface has a point nearest to the face");
                        let point = [point[0].value(), point[1].value(), point[2].value()];
                        let normal = triangle_normals[triangle];
                        let mut distance = 0.0_f64;
                        for corner in 0..4 {
                            let node = face_nodes[corner * faces + face] as usize;
                            let at = [x[node], x[nodes + node], x[2 * nodes + node]];
                            let deviation = dot(minus(at, point), normal);
                            distance = distance.max(deviation * deviation);
                        }
                        *record = [
                            point[0], point[1], point[2], normal[0], normal[1], normal[2], distance,
                        ];
                    }
                });
            }
        });
        let mut points = vec![0.0; 3 * faces];
        let mut normals = vec![0.0; 3 * faces];
        let mut distances = vec![0.0; faces];
        for (face, record) in records.iter().enumerate() {
            for d in 0..3 {
                points[d * faces + face] = record[d];
                normals[d * faces + face] = record[3 + d];
            }
            distances[face] = record[6];
        }
        (points, normals, distances)
    }
    pub fn targets(&self) -> Targets<'_> {
        Box::new(move |x| self.find(x))
    }
}

pub fn header(label: &str) {
    println!();
    println!("{label}");
    println!(
        "{:>7}{:>11}{:>11}{:>8}{:>8}{:>10}{:>10}{:>10}{:>10}{:>10}{:>10}  stopped for",
        "scale",
        "hexahedra",
        "nodes",
        "sweeps",
        "steps",
        "total s",
        "sizes s",
        "of which",
        "measure s",
        "descent s",
        "shift s"
    );
}

/// One line for a fit, and a line for each sweep if `verbose`. The seconds of the sizes include
/// those of the targets, which are said again.
pub fn row(scale: f64, elements: usize, nodes: usize, found: &Sweeps, total: f64, targets: f64) {
    let part = |index: usize| {
        found
            .sweeps
            .iter()
            .map(|sweep| sweep.seconds[index])
            .sum::<f64>()
    };
    let taken: usize = found.sweeps.iter().map(|sweep| sweep.passes).sum();
    println!(
        "{scale:>7}{elements:>11}{nodes:>11}{:>8}{taken:>8}{total:>10.3}{:>10.3}{targets:>10.3}{:>10.3}{:>10.3}{:>10.3}  {}",
        found.sweeps.len(),
        part(0),
        part(1),
        part(2),
        part(3),
        found.stop,
    );
    if std::env::var_os("VERBOSE").is_some() {
        for (index, sweep) in found.sweeps.iter().enumerate() {
            println!(
                "        sweep {index:>2}: epsilon {:.3e}, quality {:.6e}, worst {:>10.3e}, steps {:>3}, energy {:.6e}, shift {:.3e}, {:.1} ms",
                sweep.epsilon,
                sweep.quality,
                sweep.worst,
                sweep.passes,
                sweep.value,
                sweep.shift,
                1.0e3 * sweep.seconds.iter().sum::<f64>()
            );
        }
    }
}

/// The layout of a small mesh of the bone holds together: every node is in an element, the lists
/// agree with each other, and the gradient is the derivative of the energy.
#[test]
fn the_layout_of_a_bone_mesh_is_consistent() {
    let Some(target) = read() else {
        return;
    };
    let Bone {
        layout,
        coordinates,
    } = layout_of(&mesh(&target, 5.0));
    let (nodes, elements, faces) = (layout.nodes, layout.elements, layout.faces);
    assert!(elements > 1000, "{elements} hexahedra");
    assert_eq!(layout.starts[nodes] as usize, 8 * elements);
    assert!(
        layout.starts.windows(2).all(|pair| pair[0] < pair[1]),
        "a node has no element"
    );
    assert!(
        layout
            .entries
            .iter()
            .all(|&entry| (entry as usize) < 8 * elements)
    );
    assert!(
        layout
            .connectivity
            .iter()
            .all(|&node| (node as usize) < nodes)
    );
    assert!(
        layout
            .face_nodes
            .iter()
            .all(|&node| (node as usize) < nodes)
    );
    assert_eq!(layout.face_starts[nodes] as usize, 4 * faces);
    // A node is joined to the nodes that it joins, and to itself by no edge.
    let neighbors = |node: usize| {
        &layout.neighbor_entries
            [layout.neighbor_starts[node] as usize..layout.neighbor_starts[node + 1] as usize]
    };
    for node in (0..nodes).step_by(7) {
        assert!(!neighbors(node).contains(&(node as u32)));
        assert!(neighbors(node).windows(2).all(|pair| pair[0] < pair[1]));
        for &other in neighbors(node) {
            assert!(neighbors(other as usize).contains(&(node as u32)));
        }
    }
    // The gradient is the derivative of the energy.
    let targets = SurfaceTargets::new(&target, &layout);
    let backend = Emulated::new(&layout).with_targets(targets.targets());
    let x = backend.vector();
    backend.upload(&x, &coordinates);
    let typical = backend.start_sweep(&x);
    let gradient = backend.vector();
    backend.gradient(&x, &gradient);
    let gradient = backend.download(&gradient);
    let largest = gradient
        .iter()
        .fold(0.0_f64, |largest, &g| largest.max(g.abs()));
    assert!(
        largest > 0.0 && largest.is_finite(),
        "the gradient is {largest}"
    );
    let step = 1.0e-4 * typical;
    let mut checked = 0;
    for node in (0..nodes).step_by(nodes / 40 + 1) {
        for d in 0..3 {
            let index = d * nodes + node;
            let mut moved = coordinates.clone();
            moved[index] += step;
            let up = backend.objective(&std::cell::RefCell::new(moved.clone()));
            moved[index] -= 2.0 * step;
            let down = backend.objective(&std::cell::RefCell::new(moved));
            let difference = (up - down) / (2.0 * step);
            let error =
                (difference - gradient[index]).abs() / gradient[index].abs().max(1.0e-3 * largest);
            assert!(
                error < 1.0e-5,
                "node {node}, component {d}: {difference} against {}",
                gradient[index]
            );
            checked += 1;
        }
    }
    assert!(checked > 60);
}

/// The kernel that finds the targets, run on the host, finds the points that the library does, and
/// finds them again from the triangles of the last time after the nodes have moved.
#[test]
fn the_surface_kernel_finds_the_nearest_points() {
    let Some(target) = read() else {
        return;
    };
    let Bone {
        layout,
        coordinates,
    } = layout_of(&mesh(&target, 5.0));
    let (nodes, faces) = (layout.nodes, layout.faces);
    let surface = Surface::new(&soup_of(&target));
    let library = SurfaceTargets::new(&target, &layout);
    let centroid = |x: &[f64], face: usize| {
        let mut sum = [0.0; 3];
        for corner in 0..4 {
            let node = layout.face_nodes[corner * faces + face] as usize;
            for d in 0..3 {
                sum[d] += x[d * nodes + node] * 0.25;
            }
        }
        sum
    };
    let mut previous = vec![u32::MAX; faces];
    let mut x = coordinates.clone();
    let mut state = 5;
    for round in 0..3 {
        if round > 0 {
            // A small move of the nodes, so that the faces mostly keep their triangles.
            for value in x.iter_mut() {
                *value += 1.0e-3 * random(&mut state);
            }
        }
        let mut found = layout.clone();
        surface_targets_on_host(&surface, &mut found, &mut previous, &x);
        let (points, _, _) = library.find(&x);
        let mut differ = 0;
        for face in 0..faces {
            let at = centroid(&x, face);
            let distance = |points: &[f64]| {
                let d = minus(at, [0, 1, 2].map(|d| points[d * faces + face]));
                dot(d, d)
            };
            let (ours, theirs) = (distance(&found.points), distance(&points));
            assert!(
                (ours - theirs).abs() <= 1.0e-9 * theirs.max(1.0e-9),
                "round {round}, face {face}: {ours} against {theirs}"
            );
            let same = [0, 1, 2].iter().all(|&d| {
                (found.points[d * faces + face] - points[d * faces + face]).abs() < 1.0e-9
            });
            differ += usize::from(!same);
            let normal = [0, 1, 2].map(|d| found.normals[d * faces + face]);
            assert!((dot(normal, normal) - 1.0).abs() < 1.0e-9 || dot(normal, normal) == 0.0);
        }
        // Only a tie, at an edge or a corner, picks another point.
        assert!(differ * 50 <= faces, "{differ} of {faces} faces");
        assert!(
            previous
                .iter()
                .all(|&triangle| (triangle as usize) < surface.count())
        );
    }
}

/// The fit with the targets found by the kernel reaches the quality that the fit with the targets
/// of the library does, to within what the faces on an edge, which pick either triangle, change.
#[test]
fn the_fit_with_the_surface_kernel_matches_the_fit_with_the_library() {
    let Some(target) = read() else {
        return;
    };
    let Bone {
        layout,
        coordinates,
    } = layout_of(&mesh(&target, 4.0));
    let surface = Surface::new(&soup_of(&target));
    let library = SurfaceTargets::new(&target, &layout);
    let settings = Settings {
        steps: 20,
        ..Default::default()
    };
    let run = |backend: &Emulated| {
        let mut x = backend.vector();
        backend.upload(&x, &coordinates);
        fit(backend, &mut x, settings, 4, layout.epsilon)
    };
    let expected = run(&Emulated::new(&layout).with_targets(library.targets()));
    let found = run(&Emulated::new(&layout).with_surface(&surface));
    let (a, b) = (
        found.sweeps.last().unwrap(),
        expected.sweeps.last().unwrap(),
    );
    assert_eq!(found.sweeps.len(), expected.sweeps.len());
    assert!(
        (a.quality - b.quality).abs() <= 3.0e-2 * b.quality.abs().max(1.0e-3),
        "{a:?} {b:?}"
    );
    assert!(
        (a.value - b.value).abs() <= 3.0e-2 * b.value.abs().max(1.0e-3),
        "{a:?} {b:?}"
    );
}

/// The fit of a case on the threads of the host: `INTERIOR` is the mesh of the inside of `STL`.
#[test]
#[ignore]
fn case_fit_cpu() {
    let Some((target, bone, mesh)) = case() else {
        return;
    };
    let Bone {
        layout,
        coordinates,
    } = bone;
    let settings = Settings {
        steps: setting("STEPS", 100),
        ..Default::default()
    };
    let limit = setting("SWEEPS", 50);
    let (elements, nodes) = (layout.elements, layout.nodes);
    let targets = SurfaceTargets::new(&target, &layout);
    let backend = Threaded::new(layout).with_targets(targets.targets());
    header(&format!("{} threads of the host", backend.threads()));
    let mut x = backend.vector();
    backend.upload(&x, &coordinates);
    let started = Instant::now();
    let found = fit(&backend, &mut x, settings, limit, 1.0);
    let total = started.elapsed().as_secs_f64();
    row(
        0.0,
        elements,
        nodes,
        &found,
        total,
        backend.target_seconds(),
    );
    finish(mesh, nodes, &targets, &backend.download(&x));
}

/// The fit of a bone on the threads of the host, for each size of the octree in `SCALES`.
#[test]
#[ignore]
fn bone_fit_cpu() {
    let Some(target) = read() else {
        return;
    };
    let settings = Settings {
        steps: setting("STEPS", 100),
        ..Default::default()
    };
    let limit = setting("SWEEPS", 50);
    let mut printed = false;
    for scale in scales() {
        let started = Instant::now();
        let mesh = mesh(&target, scale);
        let made = started.elapsed().as_secs_f64();
        let started = Instant::now();
        let Bone {
            layout,
            coordinates,
        } = layout_of(&mesh);
        drop(mesh);
        let converted = started.elapsed().as_secs_f64();
        let (elements, nodes) = (layout.elements, layout.nodes);
        eprintln!(
            "scale {scale}: {elements} hexahedra, {nodes} nodes, {} faces; mesh {made:.1} s, layout {converted:.1} s",
            layout.faces
        );
        let targets = SurfaceTargets::new(&target, &layout);
        let backend = Threaded::new(layout).with_targets(targets.targets());
        if !printed {
            header(&format!("{} threads of the host", backend.threads()));
            printed = true;
        }
        let mut x = backend.vector();
        backend.upload(&x, &coordinates);
        let started = Instant::now();
        let found = fit(&backend, &mut x, settings, limit, 1.0);
        let total = started.elapsed().as_secs_f64();
        row(
            scale,
            elements,
            nodes,
            &found,
            total,
            backend.target_seconds(),
        );
    }
}

/// The buffer layer has an element for each face of the boundary of the inside, and a mesh that
/// the kernels take, whose gradient is the derivative of the energy at the start of its fit, where
/// the layer has no thickness.
#[test]
fn the_buffer_layer_is_a_layout() {
    let Some(target) = read() else {
        return;
    };
    let interior = mesh(&target, 5.0);
    let elements = interior.number_of_elements();
    let faces = interior.exterior_faces().len();
    let (layered, layer) = interior.buffer_layer().expect("the layer is added");
    assert_eq!(layered.number_of_elements(), elements + faces);
    assert!(!layer.is_empty());
    let Bone {
        layout,
        coordinates,
    } = layout_of(&layered);
    assert_eq!(layout.elements, elements + faces);
    assert_eq!(layout.faces, faces);
    let targets = SurfaceTargets::new(&target, &layout);
    let backend = Emulated::new(&layout).with_targets(targets.targets());
    let x = backend.vector();
    backend.upload(&x, &coordinates);
    let typical = backend.start_sweep(&x);
    assert!(typical.is_finite() && typical > 0.0);
    let (quality, worst) = backend.measure(&x);
    assert!(quality.is_finite(), "{quality}");
    // The layer starts flat, so the least determinant is zero.
    assert!(worst.abs() < 1.0e-9, "{worst}");
}

/// Writes the inside of `STL`, as the bone benchmarks mesh it at the first of `SCALES`, to the
/// Exodus file `OUTPUT`, to run `case_fit_cpu` and `case_fit_gpu` on without a case of one's own.
#[test]
#[ignore]
fn bone_interior_file() {
    let Some(target) = read() else {
        return;
    };
    let path = std::env::var_os("OUTPUT").expect("OUTPUT is the file to write");
    let scale = scales()[0];
    mesh(&target, scale)
        .write(Output::Exodus(ExodusFormat::Classic(path)))
        .expect("the mesh is written");
}
