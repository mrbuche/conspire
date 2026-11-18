#[cfg(test)]
mod test;

use super::Target;
use crate::{
    geometry::{
        Coordinate, Coordinates, Direction, DirectionsRef,
        bvh::BoundingVolumeHierarchy,
        mesh::{Mesh, Tessellation},
    },
    math::{ContractWith, Quantity, Scalar, Tensor},
    units::{Dimensionless, Length},
};
use core::array::from_fn;
use std::thread::scope;

const CREASE: Quantity = Dimensionless::of(0.866);
const DEGENERACY: Quantity = Dimensionless::of(1.0e-12);
const NORMAL_FLOOR: Quantity = Dimensionless::of(1.0e-9);
const TIE_TOLERANCE: Scalar = 1.0e-6;

pub(super) struct Oracle<'a> {
    bvh: &'a BoundingVolumeHierarchy<3>,
    coordinates: &'a Coordinates<3>,
    elements: Vec<&'a [usize]>,
    normals: DirectionsRef<'a, 3>,
    corners: Vec<[Direction<3>; 3]>,
}

impl<'a> Oracle<'a> {
    pub(super) fn new(target: &'a Tessellation) -> Self {
        let surface = target.mesh();
        let normals: Vec<&Direction<3>> = target.normals().iter().flatten().collect();
        Self {
            bvh: target.bvh(),
            coordinates: surface.coordinates(),
            corners: corner_normals(surface, &normals),
            elements: surface.connectivities().iter().flatten().collect(),
            normals: normals.into_iter().collect(),
        }
    }
    fn tied_target(
        &self,
        centroid: &Coordinate<3>,
        tolerance: Quantity<Length>,
    ) -> Option<(Coordinate<3>, Direction<3>)> {
        let ties = self
            .bvh
            .closest_points(centroid, self.coordinates, &self.elements, tolerance);
        let (nearest, index) = ties
            .iter()
            .min_by(|(a, _), (b, _)| (a - centroid).norm().total_cmp(&(b - centroid).norm()))?;
        if ties.len() == 1 {
            return Some((nearest.clone(), self.smooth_normal(*index, nearest)));
        }
        let point =
            ties.iter().map(|(point, _)| point).sum::<Coordinate<3>>() / ties.len() as Scalar;
        let normal = ties
            .iter()
            .map(|&(_, index)| &self.normals[index])
            .sum::<Direction<3>>();
        if normal.norm() > NORMAL_FLOOR {
            Some((point, normal.normalized()))
        } else {
            Some((nearest.clone(), self.smooth_normal(*index, nearest)))
        }
    }
    fn smooth_normal(&self, index: usize, point: &Coordinate<3>) -> Direction<3> {
        let facet = self.normals[index].clone();
        let triangle = self.elements[index];
        let a = &self.coordinates[triangle[0]];
        let (v0, v1, v2) = (
            &self.coordinates[triangle[1]] - a,
            &self.coordinates[triangle[2]] - a,
            point - a,
        );
        let d00 = &v0 * &v0;
        let d01 = &v0 * &v1;
        let d11 = &v1 * &v1;
        let d20 = &v2 * &v0;
        let d21 = &v2 * &v1;
        let denominator = d00 * d11 - d01 * d01;
        if denominator.norm() < (d00 * d11).norm() * DEGENERACY {
            return facet;
        }
        let beta = (d11 * d20 - d01 * d21) / denominator;
        let gamma = (d00 * d21 - d01 * d20) / denominator;
        let corners = &self.corners[index];
        let mut normal = &corners[0] * (1.0 - beta - gamma);
        normal += &corners[1] * beta;
        normal += &corners[2] * gamma;
        if normal.norm() < NORMAL_FLOOR {
            facet
        } else {
            normal.normalized()
        }
    }
    pub(super) fn targets(
        oracles: &[Self],
        owners: &[usize],
        faces: &[Vec<usize>],
        coordinates: &Coordinates<3>,
        chunk: usize,
    ) -> Result<Vec<Target>, &'static str> {
        let mut targets = vec![None; faces.len()];
        scope(|scope| {
            targets
                .chunks_mut(chunk)
                .zip(faces.chunks(chunk))
                .zip(owners.chunks(chunk))
                .for_each(|((targets, faces), owners)| {
                    scope.spawn(move || {
                        targets.iter_mut().zip(faces).zip(owners).for_each(
                            |((target, face), &owner)| {
                                let centroid = face
                                    .iter()
                                    .map(|&node| &coordinates[node])
                                    .sum::<Coordinate<3>>()
                                    / face.len() as Scalar;
                                let tolerance = face
                                    .iter()
                                    .map(|&node| (&coordinates[node] - &centroid).norm())
                                    .sum::<Quantity<Length>>()
                                    / face.len() as Scalar
                                    * TIE_TOLERANCE;
                                *target = oracles[owner].tied_target(&centroid, tolerance).map(
                                    |(point, normal)| {
                                        let distance = face
                                            .iter()
                                            .map(|&node| {
                                                let deviation =
                                                    (&coordinates[node] - &point) * &normal;
                                                deviation * deviation
                                            })
                                            .fold(Quantity::default(), Quantity::max);
                                        (point, normal, distance)
                                    },
                                );
                            },
                        )
                    });
                });
        });
        targets
            .into_iter()
            .collect::<Option<_>>()
            .ok_or("empty tessellation")
    }
}

fn corner_normals(surface: &Mesh<3>, normals: &[&Direction<3>]) -> Vec<[Direction<3>; 3]> {
    let coordinates = surface.coordinates();
    let triangles: Vec<&[usize]> = surface.connectivities().iter().flatten().collect();
    let angles: Vec<[Scalar; 3]> = triangles
        .iter()
        .map(|triangle| {
            let point = from_fn::<_, 3, _>(|corner| &coordinates[triangle[corner]]);
            from_fn(|corner| {
                let here = point[corner];
                let one = (point[(corner + 1) % 3] - here).normalized();
                let two = (point[(corner + 2) % 3] - here).normalized();
                (one * two).clamp(-1.0, 1.0).acos()
            })
        })
        .collect();
    let mut incident = vec![Vec::new(); coordinates.len()];
    triangles.iter().enumerate().for_each(|(index, triangle)| {
        (0..3).for_each(|corner| incident[triangle[corner]].push((index, corner)))
    });
    triangles
        .iter()
        .enumerate()
        .map(|(index, triangle)| {
            from_fn(|corner| {
                let own = normals[index];
                let sum = incident[triangle[corner]]
                    .iter()
                    .filter(|&&(other, _)| normals[other].contract_with(own) >= CREASE)
                    .fold(Direction::const_from([0.0; 3]), |sum, &(other, at)| {
                        sum + normals[other] * angles[other][at]
                    });
                if sum.norm() > NORMAL_FLOOR {
                    sum.normalized()
                } else {
                    own.clone()
                }
            })
        })
        .collect()
}
