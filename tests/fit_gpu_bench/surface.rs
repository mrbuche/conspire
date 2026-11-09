//! A surface of triangles, and a tree of boxes over them, to find the targets of the faces from.

use super::{Layout, Point, cross, dot, minus, shared};

/// How many triangles a leaf of the tree has at most.
const LEAF: usize = 4;

/// A surface of triangles, and a tree of boxes over them, in the form that the kernels take.
pub struct Surface {
    /// Each triangle: its three corners and its unit normal, twelve numbers, in the order of the
    /// tree.
    pub triangles: Vec<f64>,
    /// Each box: its lowest corner and its highest, six numbers, in the order that a walk down the
    /// tree visits them.
    pub bounds: Vec<f64>,
    /// For each box, the box that is next when the walk leaves out that box and what is in it.
    pub skips: Vec<u32>,
    /// For a box that has triangles, the first of them and how many, and none for the others.
    pub firsts: Vec<u32>,
    pub counts: Vec<u32>,
    /// Which of the triangles that were given each triangle of the tree is.
    pub order: Vec<usize>,
}

impl Surface {
    pub fn new(soup: &[[Point; 3]]) -> Self {
        let centers: Vec<Point> = soup
            .iter()
            .map(|triangle| {
                [0, 1, 2].map(|d| (triangle[0][d] + triangle[1][d] + triangle[2][d]) / 3.0)
            })
            .collect();
        let mut surface = Self {
            triangles: Vec::new(),
            bounds: Vec::new(),
            skips: Vec::new(),
            firsts: Vec::new(),
            counts: Vec::new(),
            order: Vec::new(),
        };
        let mut order: Vec<usize> = (0..soup.len()).collect();
        if !soup.is_empty() {
            surface.grow(soup, &centers, &mut order, 0);
        }
        surface.triangles = order
            .iter()
            .flat_map(|&index| {
                let [a, b, c] = soup[index];
                let normal = cross(minus(b, a), minus(c, a));
                let length = dot(normal, normal).sqrt();
                let normal = if length > 0.0 {
                    normal.map(|component| component / length)
                } else {
                    [0.0; 3]
                };
                [a, b, c, normal].into_iter().flatten()
            })
            .collect();
        surface.order = order;
        surface
    }
    /// The number of boxes.
    pub fn nodes(&self) -> usize {
        self.skips.len()
    }
    /// The number of triangles.
    pub fn count(&self) -> usize {
        self.triangles.len() / 12
    }
    /// Adds the box of the triangles in `order`, which start at `offset`, and the boxes below it.
    fn grow(&mut self, soup: &[[Point; 3]], centers: &[Point], order: &mut [usize], offset: usize) {
        let node = self.skips.len();
        let mut low = [f64::INFINITY; 3];
        let mut high = [f64::NEG_INFINITY; 3];
        for &index in order.iter() {
            for corner in soup[index] {
                for d in 0..3 {
                    low[d] = low[d].min(corner[d]);
                    high[d] = high[d].max(corner[d]);
                }
            }
        }
        self.bounds.extend(low);
        self.bounds.extend(high);
        self.skips.push((node + 1) as u32);
        self.firsts.push(0);
        self.counts.push(0);
        if order.len() <= LEAF {
            self.firsts[node] = offset as u32;
            self.counts[node] = order.len() as u32;
            return;
        }
        // Split at the median of the centers, along the axis that they are most spread out on.
        let spread = [0, 1, 2].map(|d| {
            let least = order
                .iter()
                .fold(f64::INFINITY, |least, &i| least.min(centers[i][d]));
            let most = order
                .iter()
                .fold(f64::NEG_INFINITY, |most, &i| most.max(centers[i][d]));
            most - least
        });
        let axis = (0..3)
            .max_by(|&a, &b| spread[a].total_cmp(&spread[b]))
            .expect("there are three axes");
        let middle = order.len() / 2;
        order.select_nth_unstable_by(middle, |&a, &b| {
            centers[a][axis].total_cmp(&centers[b][axis])
        });
        let (left, right) = order.split_at_mut(middle);
        self.grow(soup, centers, left, offset);
        self.grow(soup, centers, right, offset + middle);
        self.skips[node] = self.skips.len() as u32;
    }
}

/// The targets of the faces from a surface, found by the kernel one thread after another.
pub fn surface_targets_on_host(
    surface: &Surface,
    layout: &mut Layout,
    previous: &mut [u32],
    coordinates: &[f64],
) {
    let (nodes, faces) = (layout.nodes, layout.faces);
    let Layout {
        face_nodes,
        points,
        normals,
        distances,
        ..
    } = layout;
    for item in 0..faces {
        // SAFETY: the arrays are those of the layout and the surface.
        unsafe {
            shared::surface_targets(
                item,
                faces,
                nodes,
                surface.nodes(),
                surface.count(),
                face_nodes.as_ptr(),
                coordinates.as_ptr(),
                surface.bounds.as_ptr(),
                surface.skips.as_ptr(),
                surface.firsts.as_ptr(),
                surface.counts.as_ptr(),
                surface.triangles.as_ptr(),
                previous.as_mut_ptr(),
                points.as_mut_ptr(),
                normals.as_mut_ptr(),
                distances.as_mut_ptr(),
            )
        }
    }
}

/// The squared distance from a point to the nearest triangle, by looking at every triangle.
#[cfg(test)]
fn nearest_by_looking(soup: &[[Point; 3]], query: Point) -> f64 {
    soup.iter()
        .map(|[a, b, c]| {
            let d = minus(query, shared::closest_on_triangle(query, *a, *b, *c));
            dot(d, d)
        })
        .fold(f64::INFINITY, f64::min)
}

#[test]
fn the_tree_finds_the_nearest_triangle() {
    // A sphere of triangles whose radius varies, with a few that have no area, and points near it
    // and far from it.
    let (stacks, slices) = (30, 60);
    let mut state = 99;
    let radius = |i: usize, j: usize| 1.0 + 0.1 * ((i * 7 + j * 13) as f64 * 0.37).sin();
    let vertex = |i: usize, j: usize| -> Point {
        let theta = std::f64::consts::PI * i as f64 / stacks as f64;
        let phi = std::f64::consts::TAU * (j % slices) as f64 / slices as f64;
        [
            radius(i, j) * theta.sin() * phi.cos(),
            radius(i, j) * theta.sin() * phi.sin(),
            radius(i, j) * theta.cos(),
        ]
    };
    let mut soup: Vec<[Point; 3]> = Vec::new();
    for i in 0..stacks {
        for j in 0..slices {
            soup.push([vertex(i, j), vertex(i + 1, j), vertex(i + 1, j + 1)]);
            soup.push([vertex(i, j), vertex(i + 1, j + 1), vertex(i, j + 1)]);
        }
    }
    soup.push([vertex(3, 3), vertex(3, 3), vertex(3, 3)]);
    soup.push([vertex(5, 5), vertex(6, 5), vertex(7, 5)]);
    let surface = Surface::new(&soup);
    assert_eq!(surface.count(), soup.len());
    for node in 0..surface.nodes() {
        assert!(surface.skips[node] as usize > node);
    }
    let mut guesses = vec![
        u32::MAX,
        0,
        (soup.len() / 2) as u32,
        (soup.len() - 1) as u32,
    ];
    for _ in 0..4000 {
        let query = [0, 1, 2].map(|_| 3.0 * super::random(&mut state));
        let expected = nearest_by_looking(&soup, query);
        guesses.rotate_left(1);
        for &guess in &guesses {
            // SAFETY: the arrays are those of the surface.
            let (found, point, distance) = unsafe {
                shared::nearest_triangle(
                    query,
                    surface.nodes(),
                    surface.count(),
                    surface.bounds.as_ptr(),
                    surface.skips.as_ptr(),
                    surface.firsts.as_ptr(),
                    surface.counts.as_ptr(),
                    surface.triangles.as_ptr(),
                    guess,
                )
            };
            assert!(
                (distance - expected).abs() <= 1.0e-12 * expected.max(1.0e-12),
                "{query:?} from {guess}: {distance} against {expected}"
            );
            let d = minus(query, point);
            assert!((dot(d, d) - distance).abs() <= 1.0e-12 * distance.max(1.0e-12));
            assert!((found as usize) < soup.len());
        }
    }
}

#[test]
fn the_nearest_point_of_a_triangle_is_on_it() {
    let (a, b, c) = ([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
    // Over the face, beside an edge, past each corner, and past the long edge.
    for (query, expected) in [
        ([0.25, 0.25, 1.0], [0.25, 0.25, 0.0]),
        ([0.5, -1.0, 0.3], [0.5, 0.0, 0.0]),
        ([-1.0, -1.0, 0.0], a),
        ([2.0, -0.5, 0.0], b),
        ([-0.5, 2.0, 0.0], c),
        ([1.0, 1.0, 0.0], [0.5, 0.5, 0.0]),
    ] {
        let point = shared::closest_on_triangle(query, a, b, c);
        assert_eq!(point, expected, "from {query:?}");
    }
    // A triangle with no area has a point.
    let point = shared::closest_on_triangle([1.0, 2.0, 3.0], a, a, a);
    assert_eq!(point, a);
}
