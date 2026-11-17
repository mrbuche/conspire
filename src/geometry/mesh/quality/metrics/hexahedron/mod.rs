#[cfg(test)]
mod test;

use crate::{
    geometry::{Coordinate, Coordinates},
    math::{Quantity, Scalar, Tensor},
    units::Volume,
};

pub(crate) const CORNERS: [(usize, [usize; 3]); 8] = [
    (0, [1, 3, 4]),
    (1, [2, 0, 5]),
    (2, [3, 1, 6]),
    (3, [0, 2, 7]),
    (4, [7, 5, 0]),
    (5, [4, 6, 1]),
    (6, [5, 7, 2]),
    (7, [6, 4, 3]),
];

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

pub(super) fn edge_ratio<const D: usize>(
    element: &[usize],
    coordinates: &Coordinates<D>,
) -> Scalar {
    super::edge_ratio(&EDGES, element, coordinates)
}

pub(super) fn minimum_jacobian<const D: usize>(
    element: &[usize],
    coordinates: &Coordinates<D>,
) -> Scalar {
    super::min_jacobian(&CORNERS, element, coordinates).min(volume(element, coordinates).value())
}

pub(crate) fn minimum_scaled_jacobian<const D: usize>(
    element: &[usize],
    coordinates: &Coordinates<D>,
) -> Scalar {
    let [x1, x2, x3] = principal_axes(element, coordinates);
    let normalizer = x1.norm() * x2.norm() * x3.norm();
    let center = if normalizer > Quantity::default() {
        (triple_product(&x1, &x2, &x3) / normalizer)
            .value()
            .clamp(-1.0, 1.0)
    } else {
        0.0
    };
    super::min_scaled_jacobian(&CORNERS, element, coordinates, 1.0).min(center)
}

pub(super) fn maximum_skew<const D: usize>(
    element: &[usize],
    coordinates: &Coordinates<D>,
) -> Scalar {
    let [x1, x2, x3] = principal_axes(element, coordinates);
    [(&x1, &x2), (&x1, &x3), (&x2, &x3)]
        .into_iter()
        .map(|(u, v)| {
            let (nu, nv) = (u.norm(), v.norm());
            if nu > Quantity::default() && nv > Quantity::default() {
                ((u * v) / (nu * nv)).abs().value()
            } else {
                0.0
            }
        })
        .fold(Scalar::NEG_INFINITY, Scalar::max)
}

pub(super) fn volume<const D: usize>(
    element: &[usize],
    coordinates: &Coordinates<D>,
) -> Quantity<Volume> {
    let [x1, x2, x3] = principal_axes(element, coordinates);
    triple_product(&x1, &x2, &x3) / 64.0
}

fn principal_axes<const D: usize>(
    element: &[usize],
    coordinates: &Coordinates<D>,
) -> [Coordinate<D>; 3] {
    let p = |i: usize| &coordinates[element[i]];
    [
        (p(1) - p(0)) + (p(2) - p(3)) + (p(5) - p(4)) + (p(6) - p(7)),
        (p(3) - p(0)) + (p(2) - p(1)) + (p(7) - p(4)) + (p(6) - p(5)),
        (p(4) - p(0)) + (p(5) - p(1)) + (p(6) - p(2)) + (p(7) - p(3)),
    ]
}

fn triple_product<const D: usize>(
    x1: &Coordinate<D>,
    x2: &Coordinate<D>,
    x3: &Coordinate<D>,
) -> Quantity<Volume> {
    let c = super::cross(x2, x3);
    let zero = Quantity::default();
    let z = if D > 2 { x1[2] } else { zero };
    x1[0] * c[0] + x1[1] * c[1] + z * c[2]
}
