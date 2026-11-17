#[cfg(test)]
mod test;

use super::super::Tessellation;
use crate::{
    geometry::{Coordinate, Coordinates},
    math::{CrossProduct, FxHashMap, Quantity, Scalar, Tensor},
    units::Volume,
};
use core::f64::consts::TAU;

const D: usize = 3;

impl Tessellation {
    /// The lowest face of every shell oriented against how it is nested.
    ///
    /// A shell is a connected surface. One that is not inside any other bounds the solid and must
    /// enclose positive volume, one inside it is a cavity and must enclose negative volume, one
    /// inside that is an island and positive again, and so on. For a single shell this reduces to
    /// the sign of its enclosed volume. Meant for surfaces that are closed, manifold and do not
    /// intersect themselves, since shells are found by shared edges and nested by winding number.
    pub(super) fn inward_shells(&self) -> Vec<usize> {
        let coordinates = self.mesh().coordinates();
        let faces: Vec<[usize; 3]> = self
            .mesh()
            .connectivities()
            .iter()
            .flatten()
            .map(|face| [face[0], face[1], face[2]])
            .collect();
        let (shells, number) = shells(&faces);
        let mut volumes = vec![Quantity::<Volume>::default(); number];
        let mut lowest = vec![usize::MAX; number];
        let mut members = vec![Vec::new(); number];
        for (f, face) in faces.iter().enumerate() {
            let [a, b, c] = face.map(|node| &coordinates[node]);
            volumes[shells[f]] += b.cross(c) * a / 6.0;
            lowest[shells[f]] = lowest[shells[f]].min(f);
            members[shells[f]].push(f);
        }
        let mut inward: Vec<usize> = (0..number)
            .filter(|&shell| {
                let point = face_point(faces[lowest[shell]], coordinates);
                let depth = (0..number)
                    .filter(|&other| {
                        other != shell
                            && winding(&members[other], &faces, coordinates, &point).abs() > 0.5
                    })
                    .count();
                (volumes[shell] > Quantity::default()) != (depth % 2 == 0)
            })
            .map(|shell| lowest[shell])
            .collect();
        inward.sort_unstable();
        inward
    }
}

fn shells(faces: &[[usize; 3]]) -> (Vec<usize>, usize) {
    let mut parent: Vec<usize> = (0..faces.len()).collect();
    let mut owner = FxHashMap::<[usize; 2], usize>::default();
    for (f, face) in faces.iter().enumerate() {
        for i in 0..3 {
            let (a, b) = (face[i], face[(i + 1) % 3]);
            let edge = [a.min(b), a.max(b)];
            if let Some(&other) = owner.get(&edge) {
                let (x, y) = (find(&mut parent, f), find(&mut parent, other));
                parent[x] = y;
            } else {
                owner.insert(edge, f);
            }
        }
    }
    let mut labels = FxHashMap::<usize, usize>::default();
    let shells = (0..faces.len())
        .map(|f| {
            let root = find(&mut parent, f);
            let next = labels.len();
            *labels.entry(root).or_insert(next)
        })
        .collect();
    (shells, labels.len())
}

fn find(parent: &mut [usize], i: usize) -> usize {
    if parent[i] != i {
        parent[i] = find(parent, parent[i]);
    }
    parent[i]
}

fn face_point(face: [usize; 3], coordinates: &Coordinates<D>) -> Coordinate<D> {
    let [a, b, c] = face.map(|node| &coordinates[node]);
    (&(a + b) + c) / 3.0
}

/// The winding number of the given faces around a point, 0 outside and ±1 inside a closed shell.
fn winding(
    members: &[usize],
    faces: &[[usize; 3]],
    coordinates: &Coordinates<D>,
    point: &Coordinate<D>,
) -> Scalar {
    members
        .iter()
        .map(|&f| {
            let [a, b, c] = faces[f].map(|node| &coordinates[node] - point);
            let (la, lb, lc) = (a.norm(), b.norm(), c.norm());
            let denominator = la * lb * lc + &a * &b * lc + &b * &c * la + &c * &a * lb;
            2.0 * (&a * &b.cross(&c)).value().atan2(denominator.value())
        })
        .sum::<Scalar>()
        / (2.0 * TAU)
}
