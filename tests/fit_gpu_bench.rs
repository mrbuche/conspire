//! The energy fit of a structured grid of hexahedra, written from scratch on flat arrays, as a
//! yardstick for the fit of the library and, in the parts that follow, for the fit on a device.
//!
//! This part is the host. It makes the problem, which is a grid of hexahedra with every node free
//! to move onto a sphere, and it has the objective and the gradient of the fit energy as plain
//! loops. Nothing here uses the library: the energy is that of `src/geometry/mesh/fit`, written
//! out again, so that it can be checked against that and against a device.
//!
//! The energy is a sum over the elements of the regularized quality of each corner, weighted by
//! the size of the element, and a term that holds each node of the boundary near the faces it is on:
//!
//! - an element `e` of size `h_e` has the energy `h_e` times the sum over its corners of
//!   `trace^(3/2) / chi(eps, det)`, where `trace` is the sum of the squared lengths of the three
//!   edges at the corner, `det` is their scalar triple product, `chi` is
//!   `(det + sqrt(eps^2 + det^2)) / 2`, and `eps` is `h_e^3` times a parameter;
//! - a node `a` of size `l_a` on the faces `f` has the energy `BALANCE / l_a` times the sum over its
//!   faces of `w_f d_af^2`, where `d_af` is how far the node is from the plane of the face along its
//!   normal, and the weight `w_f` is the reciprocal of the larger of `WEIGHT_FLOOR` and the
//!   squared distance of the face over `l_a^2`.
//!
//! The target is a sphere. Each face of the boundary has the point of the sphere nearest to its
//! centroid, the normal there, and as its distance the largest squared deviation of its nodes along
//! that normal, all from the grid as it starts and fixed while the fit goes on, as the fit of the
//! library has them for the sweeps. The library holds these in a tessellation with a bounding
//! volume hierarchy, which is not part of what is timed here.
//!
//! Run alone, in release mode:
//! `cargo test --release --test fit_gpu_bench -- --ignored --nocapture --test-threads=1`
//!
//! The environment sets what is run: `CELLS` is the cells on a side of the grids, `22,46,90` by
//! default, and `REPETITIONS` is how many times each is timed, 5 by default.

use std::time::Instant;

type Point = [f64; 3];

/// How hard the boundary is held, as in the fit of the library.
const BALANCE: f64 = 2.5e3;
/// The least that the squared distance of a face over the squared size of a node counts for.
const WEIGHT_FLOOR: f64 = 0.3;
/// The grid is the cube from -1 to 1, and the sphere that it is fitted to has this radius.
const RADIUS: f64 = 1.25;
/// How far the nodes are moved from the grid, in sizes of a cell, to make the problem not trivial.
const JITTER: f64 = 0.15;
/// The corners of a hexahedron and the three nodes that they are joined to, as in the library.
const CORNERS: [(usize, [usize; 3]); 8] = [
    (0, [1, 3, 4]),
    (1, [2, 0, 5]),
    (2, [3, 1, 6]),
    (3, [0, 2, 7]),
    (4, [7, 5, 0]),
    (5, [4, 6, 1]),
    (6, [5, 7, 2]),
    (7, [6, 4, 3]),
];

fn minus(a: Point, b: Point) -> Point {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn dot(a: Point, b: Point) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross(a: Point, b: Point) -> Point {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn chi(epsilon: f64, determinant: f64) -> f64 {
    0.5 * (determinant + (epsilon * epsilon + determinant * determinant).sqrt())
}

/// A number from -1 to 1 for each call, the same for every run (SplitMix64).
fn random(state: &mut u64) -> f64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    (z >> 11) as f64 / (1_u64 << 53) as f64 * 2.0 - 1.0
}

/// The three edges that leave a corner of an element.
fn edges(element: &[u32; 8], corner: usize, adjacent: &[usize; 3], x: &[Point]) -> [Point; 3] {
    let origin = x[element[corner] as usize];
    [
        minus(x[element[adjacent[0]] as usize], origin),
        minus(x[element[adjacent[1]] as usize], origin),
        minus(x[element[adjacent[2]] as usize], origin),
    ]
}

/// The regularized quality of an element, the sum over its corners.
fn energy(element: &[u32; 8], x: &[Point], epsilon: f64) -> f64 {
    CORNERS
        .iter()
        .map(|(corner, adjacent)| {
            let e = edges(element, *corner, adjacent, x);
            let trace = dot(e[0], e[0]) + dot(e[1], e[1]) + dot(e[2], e[2]);
            trace.powf(1.5) / chi(epsilon, dot(e[0], cross(e[1], e[2])))
        })
        .sum()
}

/// How the energy of an element changes with each of its nodes, where its node `a` is the node
/// `element[a]`, in the order of the corners as the library adds them.
fn scatter(element: &[u32; 8], x: &[Point], epsilon: f64) -> [Point; 8] {
    let mut local = [[0.0; 3]; 8];
    for (corner, adjacent) in &CORNERS {
        let e = edges(element, *corner, adjacent, x);
        let trace = dot(e[0], e[0]) + dot(e[1], e[1]) + dot(e[2], e[2]);
        let determinant = dot(e[0], cross(e[1], e[2]));
        let denominator = chi(epsilon, determinant);
        let alpha = 3.0 * trace.sqrt() / denominator;
        let beta = trace.powf(1.5)
            * 0.5
            * (1.0 + determinant / (epsilon * epsilon + determinant * determinant).sqrt())
            / (denominator * denominator);
        let crosses = [cross(e[1], e[2]), cross(e[2], e[0]), cross(e[0], e[1])];
        for i in 0..3 {
            for d in 0..3 {
                local[*corner][d] += crosses[i][d] * beta - e[i][d] * alpha;
                local[adjacent[i]][d] += e[i][d] * alpha - crosses[i][d] * beta;
            }
        }
    }
    local
}

/// The weight of a face for a node of size `length`.
fn weight(distance: f64, length: f64) -> f64 {
    1.0 / (distance / (length * length)).max(WEIGHT_FLOOR)
}

/// A grid of hexahedra, with what the fit of it takes.
pub struct Problem {
    pub cells: usize,
    /// Where the nodes are when the fit starts.
    pub coordinates: Vec<Point>,
    pub elements: Vec<[u32; 8]>,
    /// The size of each node, the mean length of its edges.
    pub lengths: Vec<f64>,
    /// The size of each element, the mean of the sizes of its nodes.
    pub scales: Vec<f64>,
    /// The parameter of the regularization, which the sizes of the elements are applied to.
    pub epsilon: f64,
    /// The point, the normal and the squared distance of each face of the boundary.
    pub points: Vec<Point>,
    pub normals: Vec<Point>,
    pub distances: Vec<f64>,
    /// The faces of each node: `face_entries[face_starts[node]..face_starts[node + 1]]`.
    pub face_starts: Vec<u32>,
    pub face_entries: Vec<u32>,
    /// The corners that sit on each node, as positions among the corners of all the elements, which
    /// are `8 * element + corner`: `entries[starts[node]..starts[node + 1]]`, in the order of the
    /// elements.
    pub starts: Vec<u32>,
    pub entries: Vec<u32>,
}

impl Problem {
    pub fn new(cells: usize, epsilon: f64) -> Self {
        let side = cells + 1;
        let index = |i: usize, j: usize, k: usize| ((k * side + j) * side + i) as u32;
        let step = 2.0 / cells as f64;
        let mut state = 0x1234_5678_9ABC_DEF0;
        let mut coordinates: Vec<Point> = Vec::with_capacity(side * side * side);
        for k in 0..side {
            for j in 0..side {
                for i in 0..side {
                    let mut point = [i, j, k].map(|c| -1.0 + step * c as f64);
                    point
                        .iter_mut()
                        .for_each(|c| *c += JITTER * step * random(&mut state));
                    coordinates.push(point);
                }
            }
        }
        let mut elements = Vec::with_capacity(cells * cells * cells);
        for k in 0..cells {
            for j in 0..cells {
                for i in 0..cells {
                    elements.push([
                        index(i, j, k),
                        index(i + 1, j, k),
                        index(i + 1, j + 1, k),
                        index(i, j + 1, k),
                        index(i, j, k + 1),
                        index(i + 1, j, k + 1),
                        index(i + 1, j + 1, k + 1),
                        index(i, j + 1, k + 1),
                    ]);
                }
            }
        }
        // On a grid the nodes that an edge joins are the ones next to a node along an axis.
        let mut lengths = vec![0.0; coordinates.len()];
        for k in 0..side {
            for j in 0..side {
                for i in 0..side {
                    let node = index(i, j, k) as usize;
                    let mut sum = 0.0;
                    let mut count = 0;
                    for (axis, at) in [i, j, k].into_iter().enumerate() {
                        for neighbor in [at.checked_sub(1), (at + 1 < side).then_some(at + 1)]
                            .into_iter()
                            .flatten()
                        {
                            let mut other = [i, j, k];
                            other[axis] = neighbor;
                            let other = index(other[0], other[1], other[2]) as usize;
                            let d = minus(coordinates[other], coordinates[node]);
                            sum += dot(d, d).sqrt();
                            count += 1;
                        }
                    }
                    lengths[node] = sum / count as f64;
                }
            }
        }
        let scales = elements
            .iter()
            .map(|element| {
                element
                    .iter()
                    .map(|&node| lengths[node as usize])
                    .sum::<f64>()
                    / 8.0
            })
            .collect();
        // The faces of the boundary, as the four nodes of each.
        let mut faces: Vec<[u32; 4]> = Vec::with_capacity(6 * cells * cells);
        for a in 0..cells {
            for b in 0..cells {
                for fixed in [0, cells] {
                    faces.push([
                        index(a, b, fixed),
                        index(a + 1, b, fixed),
                        index(a + 1, b + 1, fixed),
                        index(a, b + 1, fixed),
                    ]);
                    faces.push([
                        index(a, fixed, b),
                        index(a + 1, fixed, b),
                        index(a + 1, fixed, b + 1),
                        index(a, fixed, b + 1),
                    ]);
                    faces.push([
                        index(fixed, a, b),
                        index(fixed, a + 1, b),
                        index(fixed, a + 1, b + 1),
                        index(fixed, a, b + 1),
                    ]);
                }
            }
        }
        let mut points = Vec::with_capacity(faces.len());
        let mut normals = Vec::with_capacity(faces.len());
        let mut distances = Vec::with_capacity(faces.len());
        for face in &faces {
            let mut centroid = [0.0; 3];
            for &node in face {
                (0..3).for_each(|d| centroid[d] += coordinates[node as usize][d] * 0.25);
            }
            let norm = dot(centroid, centroid).sqrt();
            let normal = centroid.map(|c| c / norm);
            let point = normal.map(|c| c * RADIUS);
            let distance = face
                .iter()
                .map(|&node| dot(minus(coordinates[node as usize], point), normal).powi(2))
                .fold(0.0, f64::max);
            points.push(point);
            normals.push(normal);
            distances.push(distance);
        }
        let nodes = coordinates.len();
        let mut face_starts = vec![0_u32; nodes + 1];
        faces
            .iter()
            .flatten()
            .for_each(|&node| face_starts[node as usize + 1] += 1);
        (0..nodes).for_each(|node| face_starts[node + 1] += face_starts[node]);
        let mut next = face_starts.clone();
        let mut face_entries = vec![0_u32; face_starts[nodes] as usize];
        for (face, nodes) in faces.iter().enumerate() {
            for &node in nodes {
                face_entries[next[node as usize] as usize] = face as u32;
                next[node as usize] += 1;
            }
        }
        let mut starts = vec![0_u32; nodes + 1];
        elements
            .iter()
            .flatten()
            .for_each(|&node| starts[node as usize + 1] += 1);
        (0..nodes).for_each(|node| starts[node + 1] += starts[node]);
        let mut next = starts.clone();
        let mut entries = vec![0_u32; starts[nodes] as usize];
        for (element, nodes) in elements.iter().enumerate() {
            for (corner, &node) in nodes.iter().enumerate() {
                entries[next[node as usize] as usize] = (8 * element + corner) as u32;
                next[node as usize] += 1;
            }
        }
        Self {
            cells,
            coordinates,
            elements,
            lengths,
            scales,
            epsilon,
            points,
            normals,
            distances,
            face_starts,
            face_entries,
            starts,
            entries,
        }
    }
    pub fn nodes(&self) -> usize {
        self.coordinates.len()
    }
    /// The energy of the elements and of the boundary, with the elements added up in order and then
    /// the nodes of the boundary.
    pub fn objective(&self, x: &[Point]) -> f64 {
        let elements: f64 = self
            .elements
            .iter()
            .zip(&self.scales)
            .map(|(element, &scale)| scale * energy(element, x, scale.powi(3) * self.epsilon))
            .sum();
        let boundary: f64 = (0..self.nodes())
            .map(|node| {
                let (first, last) = (
                    self.face_starts[node] as usize,
                    self.face_starts[node + 1] as usize,
                );
                let length = self.lengths[node];
                BALANCE / length
                    * self.face_entries[first..last]
                        .iter()
                        .map(|&face| {
                            let face = face as usize;
                            let deviation =
                                dot(minus(x[node], self.points[face]), self.normals[face]);
                            weight(self.distances[face], length) * deviation * deviation
                        })
                        .sum::<f64>()
            })
            .sum();
        elements + boundary
    }
    /// The contribution of each corner of each element to the gradient at its node, the corners of
    /// the elements one after another, as the first pass of the gradient on a device writes them.
    pub fn contributions(&self, x: &[Point], contributions: &mut [Point]) {
        for ((element, &scale), contributions) in self
            .elements
            .iter()
            .zip(&self.scales)
            .zip(contributions.chunks_mut(8))
        {
            let local = scatter(element, x, scale.powi(3) * self.epsilon);
            contributions
                .iter_mut()
                .zip(local)
                .for_each(|(contribution, point)| *contribution = point.map(|c| c * scale));
        }
    }
    /// Adds the force of the boundary to a gradient.
    fn add_boundary(&self, x: &[Point], gradient: &mut [Point]) {
        for (node, gradient) in gradient.iter_mut().enumerate() {
            let (first, last) = (
                self.face_starts[node] as usize,
                self.face_starts[node + 1] as usize,
            );
            let length = self.lengths[node];
            for &face in &self.face_entries[first..last] {
                let face = face as usize;
                let deviation = dot(minus(x[node], self.points[face]), self.normals[face]);
                let factor =
                    2.0 * BALANCE / length * weight(self.distances[face], length) * deviation;
                (0..3).for_each(|d| gradient[d] += self.normals[face][d] * factor);
            }
        }
    }
    /// The gradient, each node adding up the contributions of its corners in the order of the
    /// elements and then the force of the boundary, as the second pass on a device does.
    ///
    /// The buffer of the contributions is the caller's, `8 * elements` points, so that it is made
    /// once, as it is on a device.
    pub fn gradient_into(&self, x: &[Point], contributions: &mut [Point], gradient: &mut [Point]) {
        self.contributions(x, contributions);
        for (node, gradient) in gradient.iter_mut().enumerate() {
            let (first, last) = (self.starts[node] as usize, self.starts[node + 1] as usize);
            let mut sum = [0.0; 3];
            for &position in &self.entries[first..last] {
                (0..3).for_each(|d| sum[d] += contributions[position as usize][d]);
            }
            *gradient = sum;
        }
        self.add_boundary(x, gradient);
    }
    pub fn gradient(&self, x: &[Point]) -> Vec<Point> {
        let mut contributions = vec![[0.0; 3]; 8 * self.elements.len()];
        let mut gradient = vec![[0.0; 3]; self.nodes()];
        self.gradient_into(x, &mut contributions, &mut gradient);
        gradient
    }
    /// The same gradient, with each element adding to its nodes as it goes.
    pub fn gradient_scattered(&self, x: &[Point]) -> Vec<Point> {
        let mut gradient = vec![[0.0; 3]; self.nodes()];
        for (element, &scale) in self.elements.iter().zip(&self.scales) {
            let local = scatter(element, x, scale.powi(3) * self.epsilon);
            for (&node, point) in element.iter().zip(local) {
                (0..3).for_each(|d| gradient[node as usize][d] += point[d] * scale);
            }
        }
        self.add_boundary(x, &mut gradient);
        gradient
    }
}

fn cells() -> Vec<usize> {
    std::env::var("CELLS")
        .unwrap_or_else(|_| "22,46,90".to_string())
        .split(',')
        .filter_map(|cells| cells.trim().parse().ok())
        .collect()
}

fn repetitions() -> usize {
    std::env::var("REPETITIONS")
        .ok()
        .and_then(|repetitions| repetitions.parse().ok())
        .unwrap_or(5)
}

fn median(mut samples: Vec<f64>) -> f64 {
    samples.sort_by(f64::total_cmp);
    samples[samples.len() / 2]
}

/// The median time of a call, in milliseconds, after one that is not counted.
fn time_of(repetitions: usize, mut call: impl FnMut()) -> f64 {
    call();
    median(
        (0..repetitions)
            .map(|_| {
                let started = Instant::now();
                call();
                started.elapsed().as_secs_f64() * 1.0e3
            })
            .collect(),
    )
}

#[test]
fn the_grid_is_the_one_asked_for() {
    let cells = 4;
    let problem = Problem::new(cells, 1.0);
    assert_eq!(problem.nodes(), 125);
    assert_eq!(problem.elements.len(), 64);
    assert_eq!(problem.points.len(), 6 * cells * cells);
    assert_eq!(problem.entries.len(), 8 * 64);
    assert_eq!(problem.starts[125], 512);
    // A node on a corner of the cube is on three quads of the boundary, one along an edge of the
    // cube on four (two on each of the two sides), and one inside on none.
    assert_eq!(problem.face_starts[1] - problem.face_starts[0], 3);
    assert_eq!(problem.face_starts[2] - problem.face_starts[1], 4);
    assert_eq!(
        problem.face_starts[125 / 2 + 1] - problem.face_starts[125 / 2],
        0
    );
    // Every corner of every element is right-handed where the fit starts, and the sizes are about
    // the size of a cell.
    let step = 2.0 / cells as f64;
    for element in &problem.elements {
        for (corner, adjacent) in &CORNERS {
            let e = edges(element, *corner, adjacent, &problem.coordinates);
            assert!(dot(e[0], cross(e[1], e[2])) > 0.0);
        }
    }
    assert!(
        problem
            .lengths
            .iter()
            .all(|&length| (0.7 * step..1.3 * step).contains(&length))
    );
    // The target of a face is on the sphere, with the unit normal.
    for (point, normal) in problem.points.iter().zip(&problem.normals) {
        assert!((dot(*point, *point).sqrt() - RADIUS).abs() < 1.0e-12);
        assert!((dot(*normal, *normal) - 1.0).abs() < 1.0e-12);
    }
}

#[test]
fn a_unit_cube_has_the_energy_of_its_corners() {
    // The nodes of a hexahedron go around the bottom and then around the top.
    let x: Vec<Point> = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [0.0, 1.0, 1.0],
    ];
    let element = [0, 1, 2, 3, 4, 5, 6, 7];
    // Where the corner is a unit cube, the trace is three and the determinant is one, and with no
    // regularization chi is one.
    let expected = 8.0 * 3.0_f64.powf(1.5);
    assert!((energy(&element, &x, 0.0) - expected).abs() < 1.0e-12 * expected);
}

#[test]
fn gathering_the_gradient_is_scattering_it() {
    let problem = Problem::new(3, 1.0);
    let x = problem.coordinates.clone();
    assert_eq!(problem.gradient(&x), problem.gradient_scattered(&x));
}

#[test]
fn the_gradient_is_the_derivative_of_the_objective() {
    let problem = Problem::new(3, 1.0);
    let mut state = 42;
    // Away from where the fit starts, so that the boundary has something to pull.
    let x: Vec<Point> = problem
        .coordinates
        .iter()
        .map(|point| point.map(|c| c * 1.05 + 0.02 * random(&mut state)))
        .collect();
    let gradient = problem.gradient(&x);
    // The energy is of the order of a thousand, so a smaller step would be lost in its rounding.
    let step = 1.0e-4;
    let mut checked = 0;
    for node in (0..problem.nodes()).step_by(5) {
        for d in 0..3 {
            let (mut up, mut down) = (x.clone(), x.clone());
            up[node][d] += step;
            down[node][d] -= step;
            let difference = (problem.objective(&up) - problem.objective(&down)) / (2.0 * step);
            let error = (difference - gradient[node][d]).abs() / gradient[node][d].abs().max(1.0);
            assert!(
                error < 1.0e-6,
                "node {node}, component {d}: {difference} against {}",
                gradient[node][d]
            );
            checked += 1;
        }
    }
    assert!(checked > 30);
}

/// The time of the objective and of the gradient on the calling thread, which is the host reference.
#[test]
#[ignore]
fn host_reference() {
    println!();
    println!(
        "{:>7}{:>10}{:>11}{:>14}{:>14}{:>16}{:>16}",
        "cells",
        "nodes",
        "elements",
        "objective ms",
        "gradient ms",
        "objective us/el",
        "gradient us/el"
    );
    for cells in cells() {
        let problem = Problem::new(cells, 1.0);
        let x = problem.coordinates.clone();
        let repetitions = repetitions();
        let mut value = 0.0;
        let objective = time_of(repetitions, || {
            value = std::hint::black_box(problem.objective(&x))
        });
        let mut contributions = vec![[0.0; 3]; 8 * problem.elements.len()];
        let mut forces = vec![[0.0; 3]; problem.nodes()];
        let gradient = time_of(repetitions, || {
            problem.gradient_into(&x, &mut contributions, &mut forces);
            std::hint::black_box(&forces);
        });
        let norm = forces
            .iter()
            .map(|point| dot(*point, *point))
            .sum::<f64>()
            .sqrt();
        let elements = problem.elements.len() as f64;
        println!(
            "{cells:>7}{:>10}{:>11}{objective:>14.3}{gradient:>14.3}{:>16.4}{:>16.4}   f = {value:.9e}, |g| = {norm:.9e}",
            problem.nodes(),
            problem.elements.len(),
            objective * 1.0e3 / elements,
            gradient * 1.0e3 / elements,
        );
    }
}
