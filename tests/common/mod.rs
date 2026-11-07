//! Meshes of every kind of linear element, cut from the cells of a grid of hexahedra, for the
//! benchmarks of a device: hexahedra, tetrahedra, wedges, and pyramids, which have a node at the
//! center of each cell.
//!
//! The corners of a cell are numbered by their bits, `1` for the first axis, `2` for the second
//! and `4` for the third. The nodes of an element are ordered as the element asks, which is
//! checked from the volume that they make: the elements of the library are on the side of the
//! corners that gives them a positive Jacobian.

/// The nodes of a mesh of one kind of element, and the nodes of each of its elements.
pub struct Mesh<const N: usize> {
    pub coordinates: Vec<[f64; 3]>,
    pub connectivity: Vec<[usize; N]>,
}

/// The nodes of a grid of `cells` cells on a side.
fn lattice(cells: usize) -> Vec<[f64; 3]> {
    (0..=cells)
        .flat_map(|k| {
            (0..=cells).flat_map(move |j| {
                (0..=cells).map(move |i| {
                    [
                        i as f64 / cells as f64,
                        j as f64 / cells as f64,
                        k as f64 / cells as f64,
                    ]
                })
            })
        })
        .collect()
}

/// The sign of the volume of the tetrahedron that four points make, in the order given.
fn volume(coordinates: &[[f64; 3]], a: usize, b: usize, c: usize, d: usize) -> f64 {
    let subtract = |x: usize, y: usize| {
        [
            coordinates[x][0] - coordinates[y][0],
            coordinates[x][1] - coordinates[y][1],
            coordinates[x][2] - coordinates[y][2],
        ]
    };
    let (u, v, w) = (subtract(b, a), subtract(c, a), subtract(d, a));
    (u[1] * v[2] - u[2] * v[1]) * w[0]
        + (u[2] * v[0] - u[0] * v[2]) * w[1]
        + (u[0] * v[1] - u[1] * v[0]) * w[2]
}

/// The cells of the grid, with the function that gives the node at each corner of a cell.
fn cells(cells: usize) -> impl Iterator<Item = impl Fn(usize) -> usize> {
    let index = move |i: usize, j: usize, k: usize| (k * (cells + 1) + j) * (cells + 1) + i;
    (0..cells).flat_map(move |k| {
        (0..cells).flat_map(move |j| {
            (0..cells)
                .map(move |i| move |c: usize| index(i + (c & 1), j + ((c >> 1) & 1), k + (c >> 2)))
        })
    })
}

/// The hexahedra of a grid of `cells` cells on a side.
pub fn hexahedra(size: usize) -> Mesh<8> {
    Mesh {
        coordinates: lattice(size),
        connectivity: cells(size)
            .map(|corner| [0, 1, 3, 2, 4, 5, 7, 6].map(&corner))
            .collect(),
    }
}

/// Six tetrahedra in each cell: the paths along the edges from one corner to the opposite.
pub fn tetrahedra(size: usize) -> Mesh<4> {
    let coordinates = lattice(size);
    let mut connectivity = Vec::with_capacity(6 * size.pow(3));
    for corner in cells(size) {
        for axes in [
            [1, 2, 4],
            [1, 4, 2],
            [2, 1, 4],
            [2, 4, 1],
            [4, 1, 2],
            [4, 2, 1],
        ] {
            let mut nodes = [
                corner(0),
                corner(axes[0]),
                corner(axes[0] | axes[1]),
                corner(7),
            ];
            if volume(&coordinates, nodes[0], nodes[1], nodes[2], nodes[3]) < 0.0 {
                nodes.swap(1, 2);
            }
            connectivity.push(nodes);
        }
    }
    Mesh {
        coordinates,
        connectivity,
    }
}

/// Two wedges in each cell.
pub fn wedges(size: usize) -> Mesh<6> {
    let coordinates = lattice(size);
    let mut connectivity = Vec::with_capacity(2 * size.pow(3));
    for corner in cells(size) {
        for corners in [[0, 1, 2, 4, 5, 6], [3, 2, 1, 7, 6, 5]] {
            let mut nodes = corners.map(&corner);
            if volume(&coordinates, nodes[0], nodes[1], nodes[2], nodes[3]) < 0.0 {
                nodes.swap(1, 2);
                nodes.swap(4, 5);
            }
            connectivity.push(nodes);
        }
    }
    Mesh {
        coordinates,
        connectivity,
    }
}

/// Six pyramids in each cell, one on each face of the cell with its apex at the center of it.
pub fn pyramids(size: usize) -> Mesh<5> {
    let mut coordinates = lattice(size);
    let mut connectivity = Vec::with_capacity(6 * size.pow(3));
    for corner in cells(size) {
        let center = coordinates.len();
        let mut middle = [0.0; 3];
        (0..8).for_each(|c| (0..3).for_each(|x| middle[x] += coordinates[corner(c)][x] / 8.0));
        coordinates.push(middle);
        for base in [
            [0, 1, 3, 2],
            [4, 5, 7, 6],
            [0, 1, 5, 4],
            [2, 3, 7, 6],
            [0, 2, 6, 4],
            [1, 3, 7, 5],
        ] {
            let mut nodes = [
                corner(base[0]),
                corner(base[1]),
                corner(base[2]),
                corner(base[3]),
                center,
            ];
            if volume(&coordinates, nodes[0], nodes[1], nodes[3], nodes[4]) < 0.0 {
                nodes.swap(1, 3);
            }
            connectivity.push(nodes);
        }
    }
    Mesh {
        coordinates,
        connectivity,
    }
}
