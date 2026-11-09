//! The energy fit of a structured grid of hexahedra, written from scratch on flat arrays, as a
//! yardstick for the fit of the library and, in the parts that follow, for the fit on a device.
//!
//! This makes the problem, which is a grid of hexahedra with every node free to move onto a
//! sphere, and it has the objective and the gradient of the fit energy as plain loops, the host
//! reference. Then it has the same on a CUDA device, with the `gpu` feature. Nothing here uses the
//! library but its device: the energy is that of `src/geometry/mesh/fit`, written out again, so that
//! it can be checked against that and against the device.
//!
//! The kernels are in `kernels/fit.rs`, which `kernels/fit.sh` builds to PTX, and the arithmetic
//! that they run for each thread is in `kernels/fit_shared.rs`. The test
//! `the_kernels_run_on_the_host_are_the_reference` runs that arithmetic on the host, one thread
//! after another, as the kernels are launched, and checks it against the reference, so that the
//! layout of the data and the sums are checked without a device. The tests of the device check it
//! against the host and, when there is no device, say so and pass, unless `CONSPIRE_REQUIRE_GPU`
//! is set.
//!
//! The fit itself, the limited-memory BFGS, is in `fit_gpu_bench/solver.rs`, written once for a
//! backend that has the vectors. Run on the arithmetic of the kernels on the host it is checked
//! against a version made of plain loops, and run on the device it is checked against that too.
//! `device_sweeps` runs the whole fit as the library does, in sweeps that find the sizes, the
//! targets and the regularization again, and times it by the parts. `device_fit` times one descent with every vector on the device, whole and by the parts of a step, for the
//! two-loop recursion and for the compact form of the same inverse Hessian, which has far fewer
//! reductions to wait for.
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
//! Run alone, in release mode, the host reference and then the device:
//! `cargo test --release --test fit_gpu_bench -- host_reference --ignored --nocapture --test-threads=1`
//! `kernels/fit.sh` and then
//! `CONSPIRE_REQUIRE_GPU=1 cargo test --release -F gpu --test fit_gpu_bench -- device_reference --ignored --nocapture --test-threads=1`
//! `CONSPIRE_REQUIRE_GPU=1 cargo test --release -F gpu --test fit_gpu_bench -- device_fit --ignored --nocapture --test-threads=1`
//! `CONSPIRE_REQUIRE_GPU=1 cargo test --release -F gpu --test fit_gpu_bench -- device_sweeps --ignored --nocapture --test-threads=1`
//!
//! The environment sets what is run: `CELLS` is the cells on a side of the grids, `22,46,90` by
//! default, `STEPS` is the steps of the fit of `device_fit`, 100 by default, `SOLVER` is `two-loop`,
//! `compact` or `both` (the default) for the method of its steps, `SWEEPS` is the most sweeps of
//! `device_sweeps`, 50 by default, `VERBOSE` makes it say what each sweep did, `REPETITIONS` is how
//! many times each is timed, 5 by default, `DEVICE` is the ordinal of
//! the device, `BLOCK` is the threads in a block of a launch, 128 by default, `BATCH_THREADS` and
//! `BATCH_GROUP` are the threads of the first pass of the sums of a batch of dot products and the
//! sums that a thread of the passes after it adds up, which are chosen by the size of the vectors
//! unless they are set, and
//! `FIT_KERNELS` is
//! the PTX of the kernels, where `kernels/fit.sh` builds it by default.

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
    /// The four nodes of each face of the boundary.
    pub faces: Vec<[u32; 4]>,
    /// The nodes that an edge joins to each node: `neighbor_entries[neighbor_starts[node]..
    /// neighbor_starts[node + 1]]`.
    pub neighbor_starts: Vec<u32>,
    pub neighbor_entries: Vec<u32>,
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
        let mut neighbor_starts = Vec::with_capacity(side * side * side + 1);
        let mut neighbor_entries = Vec::with_capacity(6 * side * side * side);
        neighbor_starts.push(0_u32);
        for k in 0..side {
            for j in 0..side {
                for i in 0..side {
                    for (axis, at) in [i, j, k].into_iter().enumerate() {
                        for neighbor in [at.checked_sub(1), (at + 1 < side).then_some(at + 1)]
                            .into_iter()
                            .flatten()
                        {
                            let mut other = [i, j, k];
                            other[axis] = neighbor;
                            neighbor_entries.push(index(other[0], other[1], other[2]));
                        }
                    }
                    neighbor_starts.push(neighbor_entries.len() as u32);
                }
            }
        }
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
        let mut problem = Self {
            cells,
            lengths: vec![0.0; nodes],
            scales: vec![0.0; elements.len()],
            points: vec![[0.0; 3]; faces.len()],
            normals: vec![[0.0; 3]; faces.len()],
            distances: vec![0.0; faces.len()],
            coordinates,
            elements,
            epsilon,
            faces,
            neighbor_starts,
            neighbor_entries,
            face_starts,
            face_entries,
            starts,
            entries,
        };
        let start = problem.coordinates.clone();
        problem.refresh(&start);
        problem
    }
    /// Finds the sizes of the nodes and of the elements, and the targets of the faces, from the
    /// points `x`, as the start of a sweep of the fit does.
    pub fn refresh(&mut self, x: &[Point]) {
        for node in 0..self.nodes() {
            let (first, last) = (
                self.neighbor_starts[node] as usize,
                self.neighbor_starts[node + 1] as usize,
            );
            let mut sum = 0.0;
            for &other in &self.neighbor_entries[first..last] {
                let d = minus(x[other as usize], x[node]);
                sum += dot(d, d).sqrt();
            }
            self.lengths[node] = sum / (last - first) as f64;
        }
        for (element, scale) in self.elements.iter().zip(self.scales.iter_mut()) {
            *scale = element
                .iter()
                .map(|&node| self.lengths[node as usize])
                .sum::<f64>()
                / 8.0;
        }
        for (face, nodes) in self.faces.iter().enumerate() {
            let mut centroid = [0.0; 3];
            for &node in nodes {
                (0..3).for_each(|d| centroid[d] += x[node as usize][d] * 0.25);
            }
            let norm = dot(centroid, centroid).sqrt();
            let normal = centroid.map(|c| c / norm);
            let point = normal.map(|c| c * RADIUS);
            self.distances[face] = nodes
                .iter()
                .map(|&node| dot(minus(x[node as usize], point), normal).powi(2))
                .fold(0.0, f64::max);
            self.points[face] = point;
            self.normals[face] = normal;
        }
    }
    /// The quality of the elements, which is their energy without that of the boundary, and the
    /// least determinant of a corner over the cube of the size of its element, as the start of a
    /// sweep of the fit finds them.
    pub fn measure(&self, x: &[Point]) -> (f64, f64) {
        let mut quality = 0.0;
        let mut worst = f64::INFINITY;
        for (element, &scale) in self.elements.iter().zip(&self.scales) {
            quality += scale * energy(element, x, scale.powi(3) * self.epsilon);
            let least = CORNERS
                .iter()
                .map(|(corner, adjacent)| {
                    let e = edges(element, *corner, adjacent, x);
                    dot(e[0], cross(e[1], e[2]))
                })
                .fold(f64::INFINITY, f64::min);
            worst = worst.min(least / scale.powi(3));
        }
        (quality, worst)
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

/// The arithmetic of the kernels, for one thread at a time. The host runs it a thread after another
/// to check it, and a device runs it for all the threads at once.
#[path = "../kernels/fit_shared.rs"]
mod shared;

/// The limited-memory BFGS of the fit, for a backend that has the vectors.
#[path = "fit_gpu_bench/solver.rs"]
mod solver;

/// The fit of a bone, which needs the library to make the mesh from the surface.
#[cfg(feature = "geometry")]
#[path = "fit_gpu_bench/bone.rs"]
mod bone;

fn sqrt(x: f64) -> f64 {
    x.sqrt()
}

/// The threads of the first pass of a sum, and the values that each thread of the later passes
/// adds up.
const SUMMERS: usize = 4096;
const GROUP: usize = 64;

/// A problem as the kernels take it: values on planes, and indices of 32 bits.
#[derive(Clone)]
pub struct Layout {
    pub nodes: usize,
    pub elements: usize,
    pub faces: usize,
    pub epsilon: f64,
    pub connectivity: Vec<u32>,
    pub scales: Vec<f64>,
    pub lengths: Vec<f64>,
    pub starts: Vec<u32>,
    pub entries: Vec<u32>,
    pub face_starts: Vec<u32>,
    pub face_entries: Vec<u32>,
    /// The nodes that an edge joins to each node, as in the problem.
    pub neighbor_starts: Vec<u32>,
    pub neighbor_entries: Vec<u32>,
    /// The nodes of the faces of the boundary, `face_nodes[c * faces + face]` for the node `c`.
    pub face_nodes: Vec<u32>,
    pub points: Vec<f64>,
    pub normals: Vec<f64>,
    pub distances: Vec<f64>,
}

/// The components of points one plane after another.
fn planes(points: &[Point]) -> Vec<f64> {
    (0..3)
        .flat_map(|d| points.iter().map(move |point| point[d]))
        .collect()
}

/// The points of components that are one plane after another.
fn points_of(planes: &[f64]) -> Vec<Point> {
    let count = planes.len() / 3;
    (0..count)
        .map(|index| {
            [
                planes[index],
                planes[count + index],
                planes[2 * count + index],
            ]
        })
        .collect()
}

impl Layout {
    pub fn new(problem: &Problem) -> Self {
        let elements = problem.elements.len();
        let connectivity = (0..8)
            .flat_map(|corner| problem.elements.iter().map(move |element| element[corner]))
            .collect();
        // A corner is at `8 * element + corner` on the host and at `corner * elements + element`
        // on the planes.
        let entries = problem
            .entries
            .iter()
            .map(|&position| {
                let (element, corner) = (position as usize / 8, position as usize % 8);
                (corner * elements + element) as u32
            })
            .collect();
        Self {
            nodes: problem.nodes(),
            elements,
            faces: problem.points.len(),
            epsilon: problem.epsilon,
            connectivity,
            scales: problem.scales.clone(),
            lengths: problem.lengths.clone(),
            starts: problem.starts.clone(),
            entries,
            face_starts: problem.face_starts.clone(),
            face_entries: problem.face_entries.clone(),
            neighbor_starts: problem.neighbor_starts.clone(),
            neighbor_entries: problem.neighbor_entries.clone(),
            face_nodes: (0..4)
                .flat_map(|c| problem.faces.iter().map(move |face| face[c]))
                .collect(),
            points: planes(&problem.points),
            normals: planes(&problem.normals),
            distances: problem.distances.clone(),
        }
    }
}

/// The sum of `values` as the passes of the kernels make it, run on the host.
fn sum_by_passes(values: &[f64], summers: usize, group: usize) -> f64 {
    let mut partials = vec![0.0; summers];
    for item in 0..summers {
        // SAFETY: the values are `values.len()` and the partials are `summers`.
        unsafe {
            shared::sum_strided(
                item,
                values.len(),
                summers,
                values.as_ptr(),
                partials.as_mut_ptr(),
            )
        }
    }
    while partials.len() > 1 {
        let groups = partials.len().div_ceil(group);
        let mut sums = vec![0.0; groups];
        for item in 0..groups {
            // SAFETY: the values are `partials.len()` and the sums are `groups`.
            unsafe {
                shared::sum_columns(
                    item,
                    partials.len(),
                    group,
                    1,
                    partials.as_ptr(),
                    sums.as_mut_ptr(),
                )
            }
        }
        partials = sums;
    }
    partials[0]
}

/// The least of `values` as the passes of the kernels make it, run on the host.
fn min_by_passes(values: &[f64], threads: usize, group: usize) -> f64 {
    let mut partials = vec![0.0; threads];
    for item in 0..threads {
        // SAFETY: the values are `values.len()` and the partials are `threads`.
        unsafe {
            shared::min_strided(
                item,
                values.len(),
                threads,
                values.as_ptr(),
                partials.as_mut_ptr(),
            )
        }
    }
    while partials.len() > 1 {
        let groups = partials.len().div_ceil(group);
        let mut mins = vec![0.0; groups];
        for item in 0..groups {
            // SAFETY: the values are `partials.len()` and the minima are `groups`.
            unsafe {
                shared::min_columns(
                    item,
                    partials.len(),
                    group,
                    1,
                    partials.as_ptr(),
                    mins.as_mut_ptr(),
                )
            }
        }
        partials = mins;
    }
    partials[0]
}

/// The objective as the kernels compute it, run on the host.
fn objective_on_host(layout: &Layout, coordinates: &[f64], summers: usize, group: usize) -> f64 {
    let (nodes, elements, faces) = (layout.nodes, layout.elements, layout.faces);
    let mut energies = vec![0.0; elements + nodes];
    for item in 0..elements {
        // SAFETY: the arrays are those of the layout.
        unsafe {
            shared::element_energies(
                item,
                elements,
                nodes,
                layout.connectivity.as_ptr(),
                coordinates.as_ptr(),
                layout.scales.as_ptr(),
                layout.epsilon,
                energies.as_mut_ptr(),
            )
        }
    }
    for item in 0..nodes {
        // SAFETY: the arrays are those of the layout, and there is an energy for each node after
        // those of the elements.
        unsafe {
            shared::node_energies(
                item,
                nodes,
                elements,
                layout.face_starts.as_ptr(),
                layout.face_entries.as_ptr(),
                faces,
                layout.points.as_ptr(),
                layout.normals.as_ptr(),
                layout.distances.as_ptr(),
                coordinates.as_ptr(),
                layout.lengths.as_ptr(),
                energies.as_mut_ptr(),
            )
        }
    }
    sum_by_passes(&energies, summers, group)
}

/// The gradient as the kernels compute it, run on the host, on planes.
fn gradient_on_host(layout: &Layout, coordinates: &[f64]) -> Vec<f64> {
    let (nodes, elements, faces) = (layout.nodes, layout.elements, layout.faces);
    let mut contributions = vec![0.0; 24 * elements];
    for item in 0..elements {
        // SAFETY: the arrays are those of the layout.
        unsafe {
            shared::corner_contributions(
                item,
                elements,
                nodes,
                layout.connectivity.as_ptr(),
                coordinates.as_ptr(),
                layout.scales.as_ptr(),
                layout.epsilon,
                contributions.as_mut_ptr(),
            )
        }
    }
    let mut gradient = vec![0.0; 3 * nodes];
    for item in 0..nodes {
        // SAFETY: the arrays are those of the layout.
        unsafe {
            shared::gradient(
                item,
                nodes,
                elements,
                layout.starts.as_ptr(),
                layout.entries.as_ptr(),
                contributions.as_ptr(),
                layout.face_starts.as_ptr(),
                layout.face_entries.as_ptr(),
                faces,
                layout.points.as_ptr(),
                layout.normals.as_ptr(),
                layout.distances.as_ptr(),
                coordinates.as_ptr(),
                layout.lengths.as_ptr(),
                gradient.as_mut_ptr(),
            )
        }
    }
    gradient
}

/// The largest difference between two gradients over the largest component of the first.
fn disagreement(a: &[Point], b: &[Point]) -> f64 {
    let largest = a
        .iter()
        .flatten()
        .fold(0.0_f64, |largest, &c| largest.max(c.abs()));
    let difference = a
        .iter()
        .zip(b)
        .flat_map(|(a, b)| (0..3).map(|d| (a[d] - b[d]).abs()))
        .fold(0.0_f64, f64::max);
    difference / largest.max(1.0)
}

/// Coordinates that are away from where the fit starts, so that the boundary has something to pull.
fn moved(problem: &Problem) -> Vec<Point> {
    let mut state = 7;
    problem
        .coordinates
        .iter()
        .map(|point| point.map(|c| c * 1.05 + 0.01 * random(&mut state)))
        .collect()
}

#[test]
fn the_kernels_run_on_the_host_are_the_reference() {
    let problem = Problem::new(6, 1.0);
    let layout = Layout::new(&problem);
    let x = moved(&problem);
    let coordinates = planes(&x);
    let expected = problem.objective(&x);
    // However many threads add up the energies, the sum is the same to within the rounding.
    for summers in [1, 7, 64, SUMMERS, 100_000] {
        let objective = objective_on_host(&layout, &coordinates, summers, solver::BATCH_GROUP);
        assert!(
            (objective - expected).abs() < 1.0e-12 * expected.abs(),
            "{summers} summers: {objective} against {expected}"
        );
    }
    let gradient = points_of(&gradient_on_host(&layout, &coordinates));
    let error = disagreement(&problem.gradient(&x), &gradient);
    assert!(error < 1.0e-12, "the gradients disagree by {error}");
}

#[cfg(feature = "gpu")]
mod device {
    #[cfg(feature = "geometry")]
    use super::bone::{self, Bone, SurfaceTargets};
    use super::{
        GROUP, Layout, Point, Problem, SUMMERS, cells, disagreement, moved, planes, points_of,
        repetitions, solver::*, time_of,
    };
    use conspire::gpu::{Argument, Buffer, Device, Function, GpuError};
    use std::{
        cell::{Cell, RefCell},
        time::Instant,
    };

    /// A number that the environment sets, or what it is without.
    fn setting(name: &str, default: usize) -> usize {
        std::env::var(name)
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(default)
    }

    /// How many threads are in a block.
    fn block() -> usize {
        std::env::var("BLOCK")
            .ok()
            .and_then(|block| block.parse().ok())
            .unwrap_or(128)
    }

    fn skip<T>(why: &str) -> Option<T> {
        assert!(
            std::env::var_os("CONSPIRE_REQUIRE_GPU").is_none(),
            "CONSPIRE_REQUIRE_GPU is set and {why}"
        );
        eprintln!("skipped: {why}");
        None
    }

    /// The device and the functions of the kernels.
    struct Kernels {
        device: Device,
        corner_contributions: Function,
        element_energies: Function,
        node_energies: Function,
        gradient: Function,
        sum_strided: Function,
        sum_groups: Function,
        dot_strided: Function,
        sum_into: Function,
        times: Function,
        axpy: Function,
        scale: Function,
        scale_in_place: Function,
        subtract_alpha: Function,
        subtract_pair: Function,
        history_dots: Function,
        update_dots: Function,
        sum_columns: Function,
        combine: Function,
        scale_at: Function,
        difference_at: Function,
        node_lengths: Function,
        element_scales: Function,
        face_targets: Function,
        element_worst: Function,
        shift_values: Function,
        min_strided: Function,
        min_columns: Function,
    }

    impl Kernels {
        fn new() -> Option<Self> {
            let ordinal = std::env::var("DEVICE")
                .ok()
                .and_then(|ordinal| ordinal.parse().ok())
                .unwrap_or(0);
            let device = match Device::new(ordinal) {
                Ok(device) => device,
                Err(GpuError::Unavailable(message)) => {
                    return skip(&format!("there is no device: {message}"));
                }
                Err(error) => panic!("{error}"),
            };
            let path = std::env::var("FIT_KERNELS").unwrap_or_else(|_| {
                format!(
                    "{}/target/nvptx64-nvidia-cuda/release/examples/fit_kernels.ptx",
                    env!("CARGO_MANIFEST_DIR")
                )
            });
            let Ok(ptx) = std::fs::read_to_string(&path) else {
                return skip(&format!(
                    "the kernels are not built at {path}: run kernels/fit.sh"
                ));
            };
            let module = device
                .module_from_ptx(&ptx)
                .expect("the driver loads the kernels");
            let function = |name: &str| module.function(name).expect("the kernel is there");
            Some(Self {
                corner_contributions: function("corner_contributions"),
                element_energies: function("element_energies"),
                node_energies: function("node_energies"),
                gradient: function("gradient"),
                sum_strided: function("sum_strided"),
                sum_groups: function("sum_groups"),
                dot_strided: function("dot_strided"),
                sum_into: function("sum_into"),
                times: function("times"),
                axpy: function("axpy"),
                scale: function("scale"),
                scale_in_place: function("scale_in_place"),
                subtract_alpha: function("subtract_alpha"),
                subtract_pair: function("subtract_pair"),
                history_dots: function("history_dots"),
                update_dots: function("update_dots"),
                sum_columns: function("sum_columns"),
                combine: function("combine"),
                scale_at: function("scale_at"),
                difference_at: function("difference_at"),
                node_lengths: function("node_lengths"),
                element_scales: function("element_scales"),
                face_targets: function("face_targets"),
                element_worst: function("element_worst"),
                shift_values: function("shift_values"),
                min_strided: function("min_strided"),
                min_columns: function("min_columns"),
                device,
            })
        }
    }

    /// A problem on the device, with the buffers that the passes use.
    struct OnDevice<'a> {
        kernels: &'a Kernels,
        layout: &'a Layout,
        block: usize,
        connectivity: Buffer<u32>,
        scales: Buffer<f64>,
        lengths: Buffer<f64>,
        starts: Buffer<u32>,
        entries: Buffer<u32>,
        face_starts: Buffer<u32>,
        face_entries: Buffer<u32>,
        points: Buffer<f64>,
        normals: Buffer<f64>,
        distances: Buffer<f64>,
        coordinates: Buffer<f64>,
        contributions: Buffer<f64>,
        gradient: Buffer<f64>,
        energies: Buffer<f64>,
        partials: Buffer<f64>,
        groups: Buffer<f64>,
        total: Buffer<f64>,
        dots: Buffer<f64>,
        alphas: Buffer<f64>,
        betas: Buffer<f64>,
        /// The threads of the first pass of the sums of a batch, and the sums that a thread of a
        /// later pass adds up.
        batch_threads: usize,
        batch_group: usize,
        batch_partials: Buffer<f64>,
        /// The partial sums that the passes after the first make, but the last.
        levels: Vec<Buffer<f64>>,
        batch: Buffer<f64>,
        order: Buffer<u32>,
        coefficients: Buffer<f64>,
        neighbor_starts: Buffer<u32>,
        neighbor_entries: Buffer<u32>,
        face_nodes: Buffer<u32>,
        /// Values of the elements or of the nodes that a sweep reduces, whichever are more.
        scratch: Buffer<f64>,
        /// The least value that a reduction found.
        extreme: Buffer<f64>,
        /// The regularization, which a sweep sets.
        epsilon: Cell<f64>,
        /// What finds the targets of the faces on the host, where they are not those of a sphere,
        /// and the seconds that it and the transfers have taken.
        host_targets: Option<Targets<'a>>,
        target_seconds: Cell<f64>,
        /// The order that is on the device, so that it is not sent again.
        uploaded: RefCell<Option<Vec<usize>>>,
        /// Whether the phases of a step are waited for, to time them.
        profiling: Cell<bool>,
        started: Cell<Option<Instant>>,
        phases: RefCell<[f64; PHASES]>,
    }

    impl<'a> OnDevice<'a> {
        fn new(kernels: &'a Kernels, layout: &'a Layout) -> Self {
            let device = &kernels.device;
            let batch_threads = setting("BATCH_THREADS", automatic_threads(3 * layout.nodes));
            let batch_group = setting("BATCH_GROUP", automatic_group(batch_threads));
            assert!(batch_threads >= 2 && batch_group >= 2);
            // The passes after the first take the sums to groups of sums until there is one.
            let mut levels = Vec::new();
            let mut count = batch_threads;
            loop {
                let groups = count.div_ceil(batch_group);
                if groups == 1 {
                    break;
                }
                levels.push(device.buffer(BATCH * groups).unwrap());
                count = groups;
            }
            Self {
                kernels,
                layout,
                block: block(),
                connectivity: device.buffer_of(&layout.connectivity).unwrap(),
                scales: device.buffer_of(&layout.scales).unwrap(),
                lengths: device.buffer_of(&layout.lengths).unwrap(),
                starts: device.buffer_of(&layout.starts).unwrap(),
                entries: device.buffer_of(&layout.entries).unwrap(),
                face_starts: device.buffer_of(&layout.face_starts).unwrap(),
                face_entries: device.buffer_of(&layout.face_entries).unwrap(),
                points: device.buffer_of(&layout.points).unwrap(),
                normals: device.buffer_of(&layout.normals).unwrap(),
                distances: device.buffer_of(&layout.distances).unwrap(),
                coordinates: device.buffer(3 * layout.nodes).unwrap(),
                contributions: device.buffer(24 * layout.elements).unwrap(),
                gradient: device.buffer(3 * layout.nodes).unwrap(),
                energies: device.buffer(layout.elements + layout.nodes).unwrap(),
                partials: device.buffer(SUMMERS).unwrap(),
                groups: device.buffer(SUMMERS.div_ceil(GROUP)).unwrap(),
                total: device.buffer(1).unwrap(),
                dots: device.buffer(SLOTS).unwrap(),
                alphas: device.buffer(HISTORY + 1).unwrap(),
                betas: device.buffer(HISTORY + 1).unwrap(),
                batch_threads,
                batch_group,
                batch_partials: device.buffer(BATCH * batch_threads).unwrap(),
                levels,
                batch: device.buffer(BATCH).unwrap(),
                order: device.buffer(HISTORY).unwrap(),
                coefficients: device.buffer(1 + 2 * HISTORY).unwrap(),
                neighbor_starts: device.buffer_of(&layout.neighbor_starts).unwrap(),
                neighbor_entries: device.buffer_of(&layout.neighbor_entries).unwrap(),
                face_nodes: device.buffer_of(&layout.face_nodes).unwrap(),
                scratch: device.buffer(layout.elements.max(layout.nodes)).unwrap(),
                extreme: device.buffer(1).unwrap(),
                epsilon: Cell::new(layout.epsilon),
                host_targets: None,
                target_seconds: Cell::new(0.0),
                uploaded: RefCell::new(None),
                profiling: Cell::new(false),
                started: Cell::new(None),
                phases: RefCell::new([0.0; PHASES]),
            }
        }
        /// With the targets of the faces found by `targets` on the host, from where the nodes are
        /// that sweep.
        #[cfg_attr(not(feature = "geometry"), allow(dead_code))]
        fn with_targets(mut self, targets: Targets<'a>) -> Self {
            self.host_targets = Some(targets);
            self
        }
        fn launch(&self, function: &Function, threads: usize, arguments: &[&dyn Argument]) {
            self.kernels
                .device
                .launch(function, threads, self.block, arguments)
                .unwrap();
        }
        fn wait(&self) {
            self.kernels.device.synchronize().unwrap();
        }
        fn set_coordinates(&self, coordinates: &[f64]) {
            self.coordinates.upload(coordinates).unwrap();
        }
        /// The first pass of the gradient at `x`.
        fn corners(&self, x: &Buffer<f64>) {
            let (elements, nodes) = (self.layout.elements as u32, self.layout.nodes as u32);
            let epsilon = self.epsilon.get();
            self.launch(
                &self.kernels.corner_contributions,
                self.layout.elements,
                &[
                    &elements,
                    &nodes,
                    &self.connectivity,
                    x,
                    &self.scales,
                    &epsilon,
                    &self.contributions,
                ],
            );
        }
        /// The second pass of the gradient at `x`, into `gradient`.
        fn gather(&self, x: &Buffer<f64>, gradient: &Buffer<f64>) {
            let (nodes, elements, faces) = (
                self.layout.nodes as u32,
                self.layout.elements as u32,
                self.layout.faces as u32,
            );
            self.launch(
                &self.kernels.gradient,
                self.layout.nodes,
                &[
                    &nodes,
                    &elements,
                    &self.starts,
                    &self.entries,
                    &self.contributions,
                    &self.face_starts,
                    &self.face_entries,
                    &faces,
                    &self.points,
                    &self.normals,
                    &self.distances,
                    x,
                    &self.lengths,
                    gradient,
                ],
            );
        }
        /// The energies of the elements and of the nodes at `x`, before they are added up.
        fn energies(&self, x: &Buffer<f64>) {
            let epsilon = self.epsilon.get();
            let (elements, nodes, faces) = (
                self.layout.elements as u32,
                self.layout.nodes as u32,
                self.layout.faces as u32,
            );
            self.launch(
                &self.kernels.element_energies,
                self.layout.elements,
                &[
                    &elements,
                    &nodes,
                    &self.connectivity,
                    x,
                    &self.scales,
                    &epsilon,
                    &self.energies,
                ],
            );
            self.launch(
                &self.kernels.node_energies,
                self.layout.nodes,
                &[
                    &nodes,
                    &elements,
                    &self.face_starts,
                    &self.face_entries,
                    &faces,
                    &self.points,
                    &self.normals,
                    &self.distances,
                    x,
                    &self.lengths,
                    &self.energies,
                ],
            );
        }
        /// The passes that add the energies up: the first by many threads and then those of the
        /// batches, of one row.
        fn add_up(&self) {
            let count = (self.layout.elements + self.layout.nodes) as u32;
            let threads = self.batch_threads as u32;
            self.launch(
                &self.kernels.sum_strided,
                self.batch_threads,
                &[&count, &threads, &self.energies, &self.batch_partials],
            );
            self.reduce(1, &self.total);
        }
        fn objective_of(&self, x: &Buffer<f64>) -> f64 {
            self.energies(x);
            self.add_up();
            self.wait();
            let mut total = [0.0];
            self.total.download(&mut total).unwrap();
            total[0]
        }
        fn gradient_points(&self, x: &Buffer<f64>) -> Vec<Point> {
            self.corners(x);
            self.gather(x, &self.gradient);
            self.wait();
            points_of(&self.download(&self.gradient))
        }
    }

    /// The corrections on the device: two pools of vectors one after another.
    struct DevicePool {
        s: Buffer<f64>,
        y: Buffer<f64>,
    }

    impl OnDevice<'_> {
        /// Puts the slots of the corrections in use, oldest first, on the device, unless they are
        /// there already.
        fn set_order(&self, order: &[usize]) {
            if self.uploaded.borrow().as_deref() == Some(order) {
                return;
            }
            let mut padded = [0_u32; HISTORY];
            order
                .iter()
                .enumerate()
                .for_each(|(index, &slot)| padded[index] = slot as u32);
            self.order.upload(&padded).unwrap();
            *self.uploaded.borrow_mut() = Some(order.to_vec());
        }
        /// The passes that take the partial sums of the threads of a first pass, `rows` rows of
        /// them, to one sum for each row in `out`.
        fn reduce(&self, rows: usize, out: &Buffer<f64>) {
            self.reduce_with(&self.kernels.sum_columns, rows, out);
        }
        /// The same with the passes of `function`, which are those of the sums or those of the
        /// least values.
        fn reduce_with(&self, function: &Function, rows: usize, out: &Buffer<f64>) {
            let (group, rows_u32) = (self.batch_group as u32, rows as u32);
            let mut count = self.batch_threads;
            let mut level = 0;
            loop {
                let groups = count.div_ceil(self.batch_group);
                let input = if level == 0 {
                    &self.batch_partials
                } else {
                    &self.levels[level - 1]
                };
                let output = if groups == 1 {
                    out
                } else {
                    &self.levels[level]
                };
                let count_u32 = count as u32;
                self.launch(
                    function,
                    rows * groups,
                    &[&count_u32, &group, &rows_u32, input, output],
                );
                if groups == 1 {
                    break;
                }
                count = groups;
                level += 1;
            }
        }
        /// The passes that take the partial sums of a batch of `rows` dot products to the dot
        /// products.
        fn batch_tail(&self, rows: usize) {
            self.reduce(rows, &self.batch);
        }
    }

    impl CompactBackend for OnDevice<'_> {
        type Pool = DevicePool;
        fn pool(&self) -> DevicePool {
            let length = (HISTORY + 1) * self.count();
            DevicePool {
                s: self.kernels.device.buffer(length).unwrap(),
                y: self.kernels.device.buffer(length).unwrap(),
            }
        }
        fn store_step(&self, alpha: f64, x: &Buffer<f64>, pool: &DevicePool, slot: usize) {
            let (count, offset) = (self.count() as u32, (slot * self.count()) as u32);
            self.launch(
                &self.kernels.scale_at,
                self.count(),
                &[&count, &alpha, x, &pool.s, &offset],
            );
        }
        fn store_change(
            &self,
            new: &Buffer<f64>,
            old: &Buffer<f64>,
            pool: &DevicePool,
            slot: usize,
        ) {
            let (count, offset) = (self.count() as u32, (slot * self.count()) as u32);
            self.launch(
                &self.kernels.difference_at,
                self.count(),
                &[&count, new, old, &pool.y, &offset],
            );
        }
        fn history_dots(
            &self,
            pool: &DevicePool,
            order: &[usize],
            g: &Buffer<f64>,
            x: &Buffer<f64>,
        ) {
            self.set_order(order);
            let (count, threads, active) = (
                self.count() as u32,
                self.batch_threads as u32,
                order.len() as u32,
            );
            self.launch(
                &self.kernels.history_dots,
                self.batch_threads,
                &[
                    &count,
                    &threads,
                    &active,
                    &self.order,
                    &pool.s,
                    &pool.y,
                    g,
                    x,
                    &self.batch_partials,
                ],
            );
            self.batch_tail(ROWS_HISTORY);
        }
        fn update_dots(&self, pool: &DevicePool, order: &[usize], slot: usize) {
            self.set_order(order);
            let (count, threads, active, slot) = (
                self.count() as u32,
                self.batch_threads as u32,
                order.len() as u32,
                slot as u32,
            );
            self.launch(
                &self.kernels.update_dots,
                self.batch_threads,
                &[
                    &count,
                    &threads,
                    &active,
                    &self.order,
                    &pool.s,
                    &pool.y,
                    &slot,
                    &self.batch_partials,
                ],
            );
            self.batch_tail(ROWS_UPDATE);
        }
        fn read_batch(&self, rows: usize) -> Vec<f64> {
            self.wait();
            let mut values = [0.0; BATCH];
            self.batch.download(&mut values).unwrap();
            values[..rows].to_vec()
        }
        fn combine(
            &self,
            pool: &DevicePool,
            order: &[usize],
            coefficients: &[f64; 1 + 2 * HISTORY],
            g: &Buffer<f64>,
            d: &Buffer<f64>,
        ) {
            self.set_order(order);
            self.coefficients.upload(coefficients).unwrap();
            let (count, active) = (self.count() as u32, order.len() as u32);
            self.launch(
                &self.kernels.combine,
                self.count(),
                &[
                    &count,
                    &active,
                    &self.order,
                    &self.coefficients,
                    &pool.s,
                    &pool.y,
                    g,
                    d,
                ],
            );
        }
    }

    impl SweepBackend for OnDevice<'_> {
        fn start_sweep(&self, x: &Buffer<f64>) -> f64 {
            let (nodes, elements, faces) = (
                self.layout.nodes as u32,
                self.layout.elements as u32,
                self.layout.faces as u32,
            );
            self.launch(
                &self.kernels.node_lengths,
                self.layout.nodes,
                &[
                    &nodes,
                    &self.neighbor_starts,
                    &self.neighbor_entries,
                    x,
                    &self.lengths,
                ],
            );
            self.launch(
                &self.kernels.element_scales,
                self.layout.elements,
                &[
                    &elements,
                    &nodes,
                    &self.connectivity,
                    &self.lengths,
                    &self.scales,
                ],
            );
            if let Some(targets) = &self.host_targets {
                let started = Instant::now();
                let coordinates = self.download(x);
                let (points, normals, distances) = targets(&coordinates);
                self.points.upload(&points).unwrap();
                self.normals.upload(&normals).unwrap();
                self.distances.upload(&distances).unwrap();
                self.target_seconds
                    .set(self.target_seconds.get() + started.elapsed().as_secs_f64());
            } else {
                let radius = super::RADIUS;
                self.launch(
                    &self.kernels.face_targets,
                    self.layout.faces,
                    &[
                        &faces,
                        &nodes,
                        &self.face_nodes,
                        x,
                        &radius,
                        &self.points,
                        &self.normals,
                        &self.distances,
                    ],
                );
            }
            // The mean size of the nodes.
            let threads = self.batch_threads as u32;
            self.launch(
                &self.kernels.sum_strided,
                self.batch_threads,
                &[&nodes, &threads, &self.lengths, &self.batch_partials],
            );
            self.reduce(1, &self.total);
            self.wait();
            let mut total = [0.0];
            self.total.download(&mut total).unwrap();
            total[0] / self.layout.nodes as f64
        }
        fn measure(&self, x: &Buffer<f64>) -> (f64, f64) {
            let (nodes, elements) = (self.layout.nodes as u32, self.layout.elements as u32);
            let (epsilon, threads) = (self.epsilon.get(), self.batch_threads as u32);
            self.launch(
                &self.kernels.element_energies,
                self.layout.elements,
                &[
                    &elements,
                    &nodes,
                    &self.connectivity,
                    x,
                    &self.scales,
                    &epsilon,
                    &self.energies,
                ],
            );
            self.launch(
                &self.kernels.sum_strided,
                self.batch_threads,
                &[&elements, &threads, &self.energies, &self.batch_partials],
            );
            self.reduce(1, &self.total);
            self.launch(
                &self.kernels.element_worst,
                self.layout.elements,
                &[
                    &elements,
                    &nodes,
                    &self.connectivity,
                    x,
                    &self.scales,
                    &self.scratch,
                ],
            );
            self.launch(
                &self.kernels.min_strided,
                self.batch_threads,
                &[&elements, &threads, &self.scratch, &self.batch_partials],
            );
            self.reduce_with(&self.kernels.min_columns, 1, &self.extreme);
            self.wait();
            let (mut quality, mut worst) = ([0.0], [0.0]);
            self.total.download(&mut quality).unwrap();
            self.extreme.download(&mut worst).unwrap();
            (quality[0], worst[0])
        }
        fn set_epsilon(&self, epsilon: f64) {
            self.epsilon.set(epsilon);
        }
        fn target_seconds(&self) -> f64 {
            self.target_seconds.get()
        }
        fn shift(&self, start: &Buffer<f64>, x: &Buffer<f64>) -> f64 {
            let (nodes, threads) = (self.layout.nodes as u32, self.batch_threads as u32);
            self.launch(
                &self.kernels.shift_values,
                self.layout.nodes,
                &[&nodes, start, x, &self.lengths, &self.scratch],
            );
            self.launch(
                &self.kernels.min_strided,
                self.batch_threads,
                &[&nodes, &threads, &self.scratch, &self.batch_partials],
            );
            self.reduce_with(&self.kernels.min_columns, 1, &self.extreme);
            self.wait();
            let mut least = [0.0];
            self.extreme.download(&mut least).unwrap();
            -least[0]
        }
    }

    impl Backend for OnDevice<'_> {
        type Vector = Buffer<f64>;
        fn count(&self) -> usize {
            3 * self.layout.nodes
        }
        fn vector(&self) -> Buffer<f64> {
            self.kernels.device.buffer(self.count()).unwrap()
        }
        fn upload(&self, vector: &Buffer<f64>, planes: &[f64]) {
            vector.upload(planes).unwrap();
        }
        fn download(&self, vector: &Buffer<f64>) -> Vec<f64> {
            let mut values = vec![0.0; self.count()];
            vector.download(&mut values).unwrap();
            values
        }
        fn objective(&self, x: &Buffer<f64>) -> f64 {
            self.objective_of(x)
        }
        fn gradient(&self, x: &Buffer<f64>, gradient: &Buffer<f64>) {
            self.corners(x);
            self.gather(x, gradient);
        }
        fn axpy(&self, alpha: f64, x: &Buffer<f64>, y: &Buffer<f64>, out: &Buffer<f64>) {
            let count = self.count() as u32;
            self.launch(
                &self.kernels.axpy,
                self.count(),
                &[&count, &alpha, x, y, out],
            );
        }
        fn scale(&self, alpha: f64, x: &Buffer<f64>, out: &Buffer<f64>) {
            let count = self.count() as u32;
            self.launch(&self.kernels.scale, self.count(), &[&count, &alpha, x, out]);
        }
        fn scale_in_place(&self, alpha: f64, y: &Buffer<f64>) {
            let count = self.count() as u32;
            self.launch(
                &self.kernels.scale_in_place,
                self.count(),
                &[&count, &alpha, y],
            );
        }
        fn dot(&self, a: &Buffer<f64>, b: &Buffer<f64>, slot: usize) {
            let count = self.count() as u32;
            let (summers, group, groups, slot) = (
                SUMMERS as u32,
                GROUP as u32,
                SUMMERS.div_ceil(GROUP) as u32,
                slot as u32,
            );
            self.launch(
                &self.kernels.dot_strided,
                SUMMERS,
                &[&count, &summers, a, b, &self.partials],
            );
            self.launch(
                &self.kernels.sum_groups,
                SUMMERS.div_ceil(GROUP),
                &[&summers, &group, &self.partials, &self.groups],
            );
            self.launch(
                &self.kernels.sum_into,
                1,
                &[&groups, &self.groups, &self.dots, &slot],
            );
        }
        fn read(&self) -> [f64; SLOTS] {
            self.wait();
            let mut values = [0.0; SLOTS];
            self.dots.download(&mut values).unwrap();
            values
        }
        fn times(&self, factor: f64, slot: usize, beta: bool, index: usize) {
            let (slot, index) = (slot as u32, index as u32);
            let out = if beta { &self.betas } else { &self.alphas };
            self.launch(
                &self.kernels.times,
                1,
                &[&factor, &self.dots, &slot, out, &index],
            );
        }
        fn subtract_alpha(&self, index: usize, x: &Buffer<f64>, y: &Buffer<f64>) {
            let (count, index) = (self.count() as u32, index as u32);
            self.launch(
                &self.kernels.subtract_alpha,
                self.count(),
                &[&count, &self.alphas, &index, x, y],
            );
        }
        fn subtract_pair(&self, index: usize, x: &Buffer<f64>, y: &Buffer<f64>) {
            let (count, index) = (self.count() as u32, index as u32);
            self.launch(
                &self.kernels.subtract_pair,
                self.count(),
                &[&count, &self.alphas, &self.betas, &index, x, y],
            );
        }
        fn begin(&self, _: Phase) {
            if self.profiling.get() {
                self.wait();
                self.started.set(Some(Instant::now()));
            }
        }
        fn end(&self, phase: Phase) {
            if self.profiling.get() {
                self.wait();
                if let Some(started) = self.started.take() {
                    self.phases.borrow_mut()[phase as usize] +=
                        started.elapsed().as_secs_f64() * 1.0e3;
                }
            }
        }
    }

    #[test]
    fn the_device_agrees_with_the_host() {
        let Some(kernels) = Kernels::new() else {
            return;
        };
        let problem = Problem::new(6, 1.0);
        let layout = Layout::new(&problem);
        let on_device = OnDevice::new(&kernels, &layout);
        let x = moved(&problem);
        on_device.set_coordinates(&planes(&x));
        let expected = problem.objective(&x);
        let objective = on_device.objective_of(&on_device.coordinates);
        assert!(
            (objective - expected).abs() < 1.0e-11 * expected.abs(),
            "the objective is {objective} on the device and {expected} on the host"
        );
        let error = disagreement(
            &problem.gradient(&x),
            &on_device.gradient_points(&on_device.coordinates),
        );
        assert!(error < 1.0e-11, "the gradients disagree by {error}");
    }

    /// The fit on the device takes the steps that the plain loops on the host do.
    #[test]
    fn the_device_solver_follows_the_reference() {
        let Some(kernels) = Kernels::new() else {
            return;
        };
        let problem = Problem::new(6, 1.0);
        let layout = Layout::new(&problem);
        let on_device = OnDevice::new(&kernels, &layout);
        let typical = problem.lengths.iter().sum::<f64>() / problem.nodes() as f64;
        let settings = Settings {
            steps: 15,
            ..Default::default()
        };
        let (expected, expected_x) = reference(&problem, &problem.coordinates, typical, settings);
        let mut x = on_device.vector();
        on_device.upload(&x, &planes(&problem.coordinates));
        let trace = lbfgs(&on_device, &mut x, typical, settings);
        let (difference, same_steps) = compare(&trace, &expected);
        assert!(same_steps, "{:?} against {:?}", trace.steps, expected.steps);
        assert!(difference < 1.0e-8, "the energies differ by {difference}");
        let error = disagreement(&expected_x, &points_of(&on_device.download(&x)));
        assert!(error < 1.0e-7, "the points differ by {error}");
    }

    /// The fit with the compact form on the device takes the steps that the plain loops on the
    /// host do.
    #[test]
    fn the_device_compact_solver_follows_the_reference() {
        let Some(kernels) = Kernels::new() else {
            return;
        };
        let problem = Problem::new(6, 1.0);
        let layout = Layout::new(&problem);
        let on_device = OnDevice::new(&kernels, &layout);
        let typical = problem.lengths.iter().sum::<f64>() / problem.nodes() as f64;
        let settings = Settings {
            steps: 15,
            ..Default::default()
        };
        let (expected, expected_x) = reference(&problem, &problem.coordinates, typical, settings);
        let mut x = on_device.vector();
        on_device.upload(&x, &planes(&problem.coordinates));
        let trace = lbfgs_compact(&on_device, &mut x, typical, settings);
        let (difference, same_steps) = compare(&trace, &expected);
        assert!(same_steps, "{:?} against {:?}", trace.steps, expected.steps);
        assert!(difference < 1.0e-8, "the energies differ by {difference}");
        let error = disagreement(&expected_x, &points_of(&on_device.download(&x)));
        assert!(error < 1.0e-6, "the points differ by {error}");
    }

    /// The objective and the gradient on the device, against the host, for the sizes in `CELLS`.
    ///
    /// The kernels are timed with the coordinates already on the device, which is what a fit that
    /// keeps its state there pays for each evaluation.
    #[test]
    #[ignore]
    fn device_reference() {
        let Some(kernels) = Kernels::new() else {
            return;
        };
        println!();
        println!("{}", kernels.device.name());
        println!(
            "{:>7}{:>10}{:>11}{:>16}{:>14}{:>12}{:>12}{:>14}{:>14}",
            "cells",
            "nodes",
            "elements",
            "objective ms",
            "gradient ms",
            "corners ms",
            "gather ms",
            "host obj ms",
            "host grad ms"
        );
        for cells in cells() {
            let problem = Problem::new(cells, 1.0);
            let layout = Layout::new(&problem);
            let on_device = OnDevice::new(&kernels, &layout);
            let x = problem.coordinates.clone();
            on_device.set_coordinates(&planes(&x));
            let coordinates = &on_device.coordinates;
            let expected = problem.objective(&x);
            let objective = on_device.objective_of(coordinates);
            assert!(
                (objective - expected).abs() < 1.0e-11 * expected.abs(),
                "the objective is {objective} on the device and {expected} on the host"
            );
            let error = disagreement(
                &problem.gradient(&x),
                &on_device.gradient_points(coordinates),
            );
            assert!(error < 1.0e-11, "the gradients disagree by {error}");
            let repetitions = repetitions();
            let device_objective = time_of(repetitions, || {
                on_device.energies(coordinates);
                on_device.add_up();
                on_device.wait();
            });
            let corners = time_of(repetitions, || {
                on_device.corners(coordinates);
                on_device.wait();
            });
            let gather = time_of(repetitions, || {
                on_device.gather(coordinates, &on_device.gradient);
                on_device.wait();
            });
            let device_gradient = time_of(repetitions, || {
                on_device.corners(coordinates);
                on_device.gather(coordinates, &on_device.gradient);
                on_device.wait();
            });
            let mut contributions = vec![[0.0; 3]; 8 * problem.elements.len()];
            let mut forces = vec![[0.0; 3]; problem.nodes()];
            let host_objective = time_of(repetitions, || {
                std::hint::black_box(problem.objective(&x));
            });
            let host_gradient = time_of(repetitions, || {
                problem.gradient_into(&x, &mut contributions, &mut forces);
                std::hint::black_box(&forces);
            });
            println!(
                "{cells:>7}{:>10}{:>11}{device_objective:>16.4}{device_gradient:>14.4}{corners:>12.4}{gather:>12.4}{host_objective:>14.3}{host_gradient:>14.3}   (x{:.0}, x{:.0})",
                problem.nodes(),
                problem.elements.len(),
                host_objective / device_objective,
                host_gradient / device_gradient,
            );
        }
    }

    /// The sweeps of the fit on the device are the sweeps that the plain loops on the host make.
    #[test]
    fn the_device_sweeps_follow_the_reference() {
        let Some(kernels) = Kernels::new() else {
            return;
        };
        let mut problem = Problem::new(6, 1.0);
        let layout = Layout::new(&problem);
        let on_device = OnDevice::new(&kernels, &layout);
        let start = problem.coordinates.clone();
        let settings = Settings {
            steps: 15,
            ..Default::default()
        };
        let (expected, expected_x) = fit_reference(&mut problem, &start, settings, 3);
        let mut x = on_device.vector();
        on_device.upload(&x, &planes(&start));
        let found = fit(&on_device, &mut x, settings, 3, layout.epsilon);
        assert_eq!(found.sweeps.len(), expected.sweeps.len());
        let close = |a: f64, b: f64, tolerance: f64| (a - b).abs() <= 1.0e-6 + tolerance * b.abs();
        for (index, (a, b)) in found.sweeps.iter().zip(&expected.sweeps).enumerate() {
            assert_eq!(a.passes, b.passes, "sweep {index}: {a:?} against {b:?}");
            assert!(
                close(a.epsilon, b.epsilon, 1.0e-4),
                "sweep {index}: {a:?} against {b:?}"
            );
            assert!(
                close(a.quality, b.quality, 1.0e-6),
                "sweep {index}: {a:?} against {b:?}"
            );
            assert!(
                close(a.worst, b.worst, 1.0e-4),
                "sweep {index}: {a:?} against {b:?}"
            );
            assert!(
                close(a.value, b.value, 1.0e-5),
                "sweep {index}: {a:?} against {b:?}"
            );
            assert!(
                close(a.shift, b.shift, 1.0e-4),
                "sweep {index}: {a:?} against {b:?}"
            );
        }
        let error = disagreement(&expected_x, &points_of(&on_device.download(&x)));
        assert!(error < 1.0e-4, "the points differ by {error}");
    }

    /// The fit of a bone on the device takes the sweeps that the kernels run on the host do, with
    /// the targets found from the surface.
    #[cfg(feature = "geometry")]
    #[test]
    fn the_device_bone_fit_follows_the_emulation() {
        let Some(kernels) = Kernels::new() else {
            return;
        };
        let Some(target) = bone::read() else {
            return;
        };
        let Bone {
            layout,
            coordinates,
        } = bone::layout_of(&bone::mesh(&target, 3.0));
        let targets = SurfaceTargets::new(&target, &layout);
        let settings = Settings {
            steps: 15,
            ..Default::default()
        };
        let emulated = Emulated::new(&layout).with_targets(targets.targets());
        let mut expected_x = emulated.vector();
        emulated.upload(&expected_x, &coordinates);
        let expected = fit(&emulated, &mut expected_x, settings, 3, layout.epsilon);
        let on_device = OnDevice::new(&kernels, &layout).with_targets(targets.targets());
        let mut x = on_device.vector();
        on_device.upload(&x, &coordinates);
        let found = fit(&on_device, &mut x, settings, 3, layout.epsilon);
        assert_eq!(found.sweeps.len(), expected.sweeps.len());
        let close = |a: f64, b: f64, tolerance: f64| (a - b).abs() <= 1.0e-6 + tolerance * b.abs();
        for (index, (a, b)) in found.sweeps.iter().zip(&expected.sweeps).enumerate() {
            let at = format!("sweep {index}: {a:?} against {b:?}");
            assert_eq!(a.passes, b.passes, "{at}");
            assert!(close(a.epsilon, b.epsilon, 1.0e-4), "{at}");
            assert!(close(a.quality, b.quality, 1.0e-6), "{at}");
            assert!(close(a.worst, b.worst, 1.0e-4), "{at}");
            assert!(close(a.value, b.value, 1.0e-5), "{at}");
            assert!(close(a.shift, b.shift, 1.0e-4), "{at}");
        }
    }

    /// The fit of a bone on the device, for each size of the octree in `SCALES`: the mesh is made on
    /// the host, and each sweep finds its targets from the surface on the host and sends them over.
    #[cfg(feature = "geometry")]
    #[test]
    #[ignore]
    fn bone_fit_gpu() {
        let Some(kernels) = Kernels::new() else {
            return;
        };
        let Some(target) = bone::read() else {
            return;
        };
        let settings = Settings {
            steps: bone::setting("STEPS", 100),
            ..Default::default()
        };
        let limit = bone::setting("SWEEPS", 50);
        bone::header(&format!("{}, block {}", kernels.device.name(), block()));
        for scale in bone::scales() {
            let started = Instant::now();
            let mesh = bone::mesh(&target, scale);
            let made = started.elapsed().as_secs_f64();
            let started = Instant::now();
            let Bone {
                layout,
                coordinates,
            } = bone::layout_of(&mesh);
            drop(mesh);
            let converted = started.elapsed().as_secs_f64();
            let (elements, nodes) = (layout.elements, layout.nodes);
            eprintln!(
                "scale {scale}: {elements} hexahedra, {nodes} nodes, {} faces; mesh {made:.1} s, layout {converted:.1} s",
                layout.faces
            );
            let targets = SurfaceTargets::new(&target, &layout);
            let started = Instant::now();
            let on_device = OnDevice::new(&kernels, &layout).with_targets(targets.targets());
            let mut x = on_device.vector();
            on_device.upload(&x, &coordinates);
            on_device.wait();
            eprintln!(
                "        on the device in {:.1} s; batch threads {}, groups of {}",
                started.elapsed().as_secs_f64(),
                on_device.batch_threads,
                on_device.batch_group
            );
            let started = Instant::now();
            let found = fit(&on_device, &mut x, settings, limit, layout.epsilon);
            on_device.wait();
            let total = started.elapsed().as_secs_f64();
            bone::row(
                scale,
                elements,
                nodes,
                &found,
                total,
                on_device.target_seconds(),
            );
        }
    }

    /// The whole fit on the device: every sweep, with the sizes, the targets, the quality and the
    /// regularization found on the device, and the descents.
    ///
    /// `SWEEPS` is the most sweeps, 50 by default as in the library, `STEPS` the most steps of a
    /// sweep, 100 by default, and `VERBOSE` says what each sweep did. Only the targets are not
    /// what the library does: the library finds them with a bounding volume hierarchy on the host,
    /// here the point of the sphere nearest to the face is found on the device.
    #[test]
    #[ignore]
    fn device_sweeps() {
        let Some(kernels) = Kernels::new() else {
            return;
        };
        let steps = setting("STEPS", 100);
        let limit = setting("SWEEPS", 50);
        let verbose = std::env::var_os("VERBOSE").is_some();
        println!();
        println!("{}", kernels.device.name());
        println!("block {}", block());
        println!(
            "{:>7}{:>10}{:>11}{:>8}{:>8}{:>10}{:>10}{:>10}{:>10}{:>10}  stopped for",
            "cells",
            "nodes",
            "elements",
            "sweeps",
            "steps",
            "total s",
            "sizes s",
            "measure s",
            "descent s",
            "shift s"
        );
        for cells in cells() {
            let problem = Problem::new(cells, 1.0);
            let layout = Layout::new(&problem);
            let on_device = OnDevice::new(&kernels, &layout);
            println!(
                "        batch threads {}, groups of {}",
                on_device.batch_threads, on_device.batch_group
            );
            let mut x = on_device.vector();
            on_device.upload(&x, &planes(&problem.coordinates));
            on_device.wait();
            let settings = Settings {
                steps,
                ..Default::default()
            };
            let started = Instant::now();
            let found = fit(&on_device, &mut x, settings, limit, layout.epsilon);
            on_device.wait();
            let total = started.elapsed().as_secs_f64();
            let part = |index: usize| {
                found
                    .sweeps
                    .iter()
                    .map(|sweep| sweep.seconds[index])
                    .sum::<f64>()
            };
            let taken: usize = found.sweeps.iter().map(|sweep| sweep.passes).sum();
            println!(
                "{cells:>7}{:>10}{:>11}{:>8}{taken:>8}{total:>10.3}{:>10.3}{:>10.3}{:>10.3}{:>10.3}  {}",
                problem.nodes(),
                problem.elements.len(),
                found.sweeps.len(),
                part(0),
                part(1),
                part(2),
                part(3),
                found.stop,
            );
            if verbose {
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
    }

    /// The fit on the device, all of it: the steps of the method with every vector on the device.
    ///
    /// `STEPS` is how many steps to take, 100 by default, which is what a sweep of the fit of the
    /// library takes at most, and the run does not stop for having converged. `SOLVER` is
    /// `two-loop`, `compact` or `both`, the default. It is timed whole, and then again with each
    /// part of a step waited for, which adds the time of the waiting. For the compact form,
    /// "norms" is the first batched pass of dot products, "two-loop" the solve on the host and
    /// the pass that makes the direction, and "history" the second batched pass.
    #[test]
    #[ignore]
    fn device_fit() {
        let Some(kernels) = Kernels::new() else {
            return;
        };
        let steps = std::env::var("STEPS")
            .ok()
            .and_then(|steps| steps.parse().ok())
            .unwrap_or(100);
        let solvers: Vec<&str> = match std::env::var("SOLVER").as_deref() {
            Ok("two-loop") => vec!["two-loop"],
            Ok("compact") => vec!["compact"],
            _ => vec!["two-loop", "compact"],
        };
        println!();
        println!("{}", kernels.device.name());
        println!("block {}", block());
        println!(
            "{:>7}{:>10}{:>11}{:>10}{:>7}{:>7}{:>12}{:>10}{:>16}{:>16}",
            "cells",
            "nodes",
            "elements",
            "solver",
            "steps",
            "evals",
            "total ms",
            "ms/step",
            "energy at start",
            "energy at end"
        );
        for cells in cells() {
            let problem = Problem::new(cells, 1.0);
            let layout = Layout::new(&problem);
            let on_device = OnDevice::new(&kernels, &layout);
            println!(
                "        batch threads {}, groups of {}",
                on_device.batch_threads, on_device.batch_group
            );
            let typical = problem.lengths.iter().sum::<f64>() / problem.nodes() as f64;
            let settings = Settings {
                steps,
                convergence: 0.0,
            };
            let start = planes(&problem.coordinates);
            for &solver in &solvers {
                let mut x = on_device.vector();
                let mut run = || {
                    on_device.upload(&x, &start);
                    on_device.wait();
                    let started = Instant::now();
                    let trace = if solver == "compact" {
                        lbfgs_compact(&on_device, &mut x, typical, settings)
                    } else {
                        lbfgs(&on_device, &mut x, typical, settings)
                    };
                    on_device.wait();
                    (trace, started.elapsed().as_secs_f64() * 1.0e3)
                };
                let (trace, _) = run();
                let total = super::median((0..repetitions()).map(|_| run().1).collect());
                on_device.profiling.set(true);
                run();
                on_device.profiling.set(false);
                let taken = trace.values.len();
                println!(
                    "{cells:>7}{:>10}{:>11}{solver:>10}{taken:>7}{:>7}{total:>12.3}{:>10.4}{:>16.9e}{:>16.9e}",
                    problem.nodes(),
                    problem.elements.len(),
                    trace.evaluations,
                    total / trace.passes.max(1) as f64,
                    trace.initial,
                    trace.values.last().copied().unwrap_or(trace.initial),
                );
                let phases = *on_device.phases.borrow();
                let waited: f64 = phases.iter().sum();
                let line: Vec<String> = Phase::ALL
                    .iter()
                    .map(|phase| {
                        let time = phases[*phase as usize];
                        format!(
                            "{} {:.4} ms ({:.0}%)",
                            phase.name(),
                            time / trace.passes.max(1) as f64,
                            100.0 * time / waited
                        )
                    })
                    .collect();
                println!(
                    "        per step, each part waited for: {}",
                    line.join(", ")
                );
                *on_device.phases.borrow_mut() = [0.0; PHASES];
            }
        }
    }
}
