#[cfg(test)]
mod test;

use crate::{
    geometry::{
        Coordinate, Coordinates, Direction, DirectionsRef,
        bvh::BoundingVolumeHierarchy,
        mesh::{
            Connectivity, Mesh, Tessellation,
            quality::metrics::{chi, hexahedron, pyramid, regularized, tetrahedron, wedge},
        },
    },
    math::{
        ContractWith, CrossProduct, FxHashSet, Quantity, Reference, Scalar, Tensor, TensorRank1,
        TensorRank1List, TensorRank1Vec,
        optimize::{Armijo, LimitedMemoryBFGS, Tolerances},
    },
    units::{Area, Dimensionless, Length, ReciprocalLength},
};
use alloc::collections::VecDeque;
use core::{array::from_fn, cell::RefCell, mem::replace, mem::take};
use std::{
    sync::Mutex,
    thread::{available_parallelism, scope},
};

type EdgeList = TensorRank1List<3, Reference, 3>;
type Entry = TensorRank1<3, Reference, Dimensionless>;
type Slope = TensorRank1<3, Reference, ReciprocalLength>;
type Gradient = TensorRank1Vec<3, Reference, Dimensionless>;
type Target = (Coordinate<3>, Direction<3>, Quantity<Area>);

type CornerTable = [(usize, [usize; 3])];

const ARMIJO: Scalar = 1.0e-4;
const BACKTRACKS: usize = 32;
const BALANCE: Scalar = 2.5e3;
const CONVERGENCE: Scalar = 1.0e-5;
const DEGENERACY: Quantity = Dimensionless::of(1.0e-12);
const EPSILON_FLOOR: Scalar = 1.0e-12;
const HISTORY: usize = 8;
const ITERATIONS: usize = 100;
const NORMAL_FLOOR: Quantity = Dimensionless::of(1.0e-9);
const RELAXATION: Scalar = 0.1;
const CREASE: Quantity = Dimensionless::of(0.866);
const STAGNATION: Scalar = 5.0e-4;
const SWEEPS: usize = 50;
const TIE_TOLERANCE: Scalar = 1.0e-6;
const TOLERANCE: Scalar = 1.0e-3;
const WEIGHT_FLOOR: Quantity = Dimensionless::of(0.3);
const WINDOW: usize = 3;
const PLACE_GRAIN: usize = 1 << 14;

/// The parts of a fit that are timed when profiling.
#[derive(Clone, Copy)]
#[cfg_attr(not(test), allow(dead_code))]
enum Part {
    Assign,
    Derivative,
    Descent,
    Measure,
    Objective,
    Sizes,
    Targets,
}

#[cfg(test)]
pub(crate) mod profile {
    use super::Part;
    use core::sync::atomic::{AtomicU64, Ordering::Relaxed};
    pub(crate) const PARTS: usize = 7;
    pub(crate) static NANOSECONDS: [AtomicU64; PARTS] = [const { AtomicU64::new(0) }; PARTS];
    pub(crate) static CALLS: [AtomicU64; PARTS] = [const { AtomicU64::new(0) }; PARTS];
    pub(crate) static UNKNOWNS: AtomicU64 = AtomicU64::new(0);
    pub(crate) static STEPS: AtomicU64 = AtomicU64::new(0);
    pub(crate) fn reset() {
        NANOSECONDS
            .iter()
            .chain(&CALLS)
            .for_each(|c| c.store(0, Relaxed));
        UNKNOWNS.store(0, Relaxed);
        STEPS.store(0, Relaxed);
    }
    pub(super) fn record(part: Part, started: std::time::Instant) {
        NANOSECONDS[part as usize].fetch_add(started.elapsed().as_nanos() as u64, Relaxed);
        CALLS[part as usize].fetch_add(1, Relaxed);
    }
}

#[cfg(test)]
fn timed<T>(part: Part, work: impl FnOnce() -> T) -> T {
    let started = std::time::Instant::now();
    let result = work();
    profile::record(part, started);
    result
}

#[cfg(not(test))]
#[inline(always)]
fn timed<T>(_: Part, work: impl FnOnce() -> T) -> T {
    work()
}

/// Where the contributions of the corners of the elements go: the entries of a node are the
/// positions, in the order of the elements, of the contributions of the corners that sit on it.
struct Adjacency {
    /// The position of the first corner of each tracked element, and the number of corners.
    offsets: Vec<usize>,
    /// The first entry of each unknown, and the number of entries.
    starts: Vec<usize>,
    /// The positions of the corners that sit on the unknowns, one run to an unknown.
    entries: Vec<usize>,
}

fn adjacency(
    elements: &[(&'static CornerTable, Vec<usize>)],
    tracked: &[usize],
    slot: &[Option<usize>],
    unknowns: usize,
) -> Adjacency {
    let mut offsets = Vec::with_capacity(tracked.len() + 1);
    let mut corners = 0;
    tracked.iter().for_each(|&element| {
        offsets.push(corners);
        corners += elements[element].1.len()
    });
    offsets.push(corners);
    let mut starts = vec![0; unknowns + 1];
    tracked.iter().for_each(|&element| {
        elements[element]
            .1
            .iter()
            .filter_map(|&node| slot[node])
            .for_each(|index| starts[index + 1] += 1)
    });
    (0..unknowns).for_each(|index| starts[index + 1] += starts[index]);
    let mut next = starts.clone();
    let mut entries = vec![0; starts[unknowns]];
    tracked.iter().enumerate().for_each(|(position, &element)| {
        elements[element]
            .1
            .iter()
            .enumerate()
            .for_each(|(corner, &node)| {
                if let Some(index) = slot[node] {
                    entries[next[index]] = offsets[position] + corner;
                    next[index] += 1
                }
            })
    });
    Adjacency {
        offsets,
        starts,
        entries,
    }
}

fn solver(typical: Quantity<Length>, residual: Scalar) -> LimitedMemoryBFGS<Armijo, Length> {
    LimitedMemoryBFGS {
        abs_tol: Tolerances {
            constraint: 0.0,
            residual,
        },
        history: HISTORY,
        line_search: Armijo {
            control: ARMIJO,
            cut_back: 0.5,
            max_steps: BACKTRACKS,
        },
        max_steps: ITERATIONS,
        rel_tol: None,
        typical,
    }
}

/// Constraint on how fitted nodes approach the target surface.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Fitting {
    /// The nodes settle wherever the quality and fit energies balance.
    Soft,
    /// The nodes settle as above, but the boundary is then projected onto the
    /// surface, after which the rest relax.
    Snap,
}

/// Which nodes an energy fit is free to move.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Freedom {
    /// Every node of the mesh.
    Whole,
    /// The boundary nodes and their immediate neighbours only.
    Shell,
}

struct Oracle<'a> {
    bvh: &'a BoundingVolumeHierarchy<3>,
    coordinates: &'a Coordinates<3>,
    elements: Vec<&'a [usize]>,
    normals: DirectionsRef<'a, 3>,
    corners: Vec<[Direction<3>; 3]>,
}

struct Sweep<'a> {
    adjacency: &'a Adjacency,
    contributions: &'a Mutex<Vec<Entry>>,
    element_chunk: usize,
    elements: &'a [(&'static CornerTable, Vec<usize>)],
    epsilon: Scalar,
    lengths: Vec<Quantity<Length>>,
    node_chunk: usize,
    node_faces: &'a [Vec<usize>],
    nodes: &'a [usize],
    scales: Vec<Quantity<Length>>,
    sorted: &'a [(usize, usize)],
    targets: Vec<Target>,
    threads: usize,
    tracked: &'a [usize],
    unknowns: usize,
}

impl Mesh<3> {
    /// Deforms the mesh onto the target through energy fitting.
    pub fn inflate(
        &mut self,
        target: &Tessellation,
        freedom: Freedom,
        fitting: Fitting,
    ) -> Result<(), &'static str> {
        let mut boundary: Vec<usize> = self.exterior_faces().into_iter().flatten().collect();
        boundary.sort_unstable();
        boundary.dedup();
        let free = match freedom {
            Freedom::Whole => (0..self.number_of_nodes()).collect(),
            Freedom::Shell => {
                let neighbors = self.node_node_connectivity();
                let mut nodes: Vec<usize> = boundary
                    .iter()
                    .flat_map(|&node| neighbors[node].iter().copied())
                    .chain(boundary.iter().copied())
                    .collect();
                nodes.sort_unstable();
                nodes.dedup();
                nodes
            }
        };
        self.fit_to(&free, &boundary, target, fitting)
    }
    pub(crate) fn fit_to(
        &mut self,
        free: &[usize],
        boundary: &[usize],
        target: &Tessellation,
        fitting: Fitting,
    ) -> Result<(), &'static str> {
        self.fit(free, target)?;
        if let Fitting::Snap = fitting {
            self.project(target, boundary)?;
            let pinned: FxHashSet<usize> = boundary.iter().copied().collect();
            let interior: Vec<usize> = free
                .iter()
                .copied()
                .filter(|node| !pinned.contains(node))
                .collect();
            self.fit(&interior, target)?;
        }
        Ok(())
    }
    pub(crate) fn project(
        &mut self,
        target: &Tessellation,
        nodes: &[usize],
    ) -> Result<(), &'static str> {
        let surface = target.mesh();
        let surface_coordinates = surface.coordinates();
        let elements: Vec<&[usize]> = surface.connectivities().iter().flatten().collect();
        let bvh = target.bvh();
        let coordinates = &mut self.coordinates;
        nodes.iter().try_for_each(|&node| {
            let (point, _) = bvh
                .closest_point(&coordinates[node], surface_coordinates, &elements)
                .ok_or("empty tessellation")?;
            coordinates[node] = point;
            Ok(())
        })
    }
    pub(crate) fn fit(
        &mut self,
        nodes: &[usize],
        target: &Tessellation,
    ) -> Result<(), &'static str> {
        let mut elements: Vec<(&'static CornerTable, Vec<usize>)> = Vec::new();
        for block in self.iter() {
            let corners: &'static CornerTable = match block {
                Connectivity::Hexahedral(_) => &hexahedron::CORNERS,
                Connectivity::Tetrahedral(_) => &tetrahedron::CORNERS,
                Connectivity::Pyramidal(_) => &pyramid::CORNERS,
                Connectivity::Wedge(_) => &wedge::CORNERS,
                _ => return Err("fit requires hexahedra, tetrahedra, pyramids or wedges"),
            };
            elements.extend(block.iter().map(|element| (corners, element.to_vec())));
        }
        let oracle = Oracle::new(target);
        let number_of_nodes = self.number_of_nodes();
        let mut free = vec![false; number_of_nodes];
        nodes.iter().for_each(|&node| free[node] = true);
        let node_elements = self.node_element_connectivity().to_vec();
        let tracked: Vec<usize> = {
            let mut seen = vec![false; elements.len()];
            nodes
                .iter()
                .flat_map(|&node| node_elements[node].iter().copied())
                .filter(|&element| !replace(&mut seen[element], true))
                .collect()
        };
        let faces: Vec<Vec<usize>> = self
            .exterior_faces()
            .into_iter()
            .filter(|face| face.iter().any(|&node| free[node]))
            .collect();
        let mut node_faces = vec![Vec::new(); number_of_nodes];
        faces.iter().enumerate().for_each(|(index, face)| {
            face.iter().for_each(|&node| {
                if free[node] {
                    node_faces[node].push(index)
                }
            })
        });
        let neighbors = self.node_node_connectivity().to_vec();
        let threads = available_parallelism().map_or(1, |threads| threads.get());
        let face_chunk = faces.len().div_ceil(threads).max(1);
        let element_chunk = tracked.len().div_ceil(threads).max(1);
        let node_chunk = nodes.len().div_ceil(threads).max(1);
        let coordinates = &mut self.coordinates;
        let mut slot = vec![None; number_of_nodes];
        nodes
            .iter()
            .enumerate()
            .for_each(|(index, &node)| slot[node] = Some(index));
        let mut sorted: Vec<(usize, usize)> = nodes.iter().copied().zip(0..).collect();
        sorted.sort_by_key(|&(node, _)| node);
        let unknowns = nodes.len();
        let adjacency = adjacency(&elements, &tracked, &slot, unknowns);
        let contributions: Mutex<Vec<Entry>> = Mutex::new(vec![
            TensorRank1::const_from([0.0; 3]);
            adjacency.offsets[tracked.len()]
        ]);
        let mut epsilon: Scalar = 1.0;
        let mut previous = Quantity::<Length>::new(Scalar::INFINITY);
        let mut window = VecDeque::<Quantity<Length>>::with_capacity(WINDOW);
        for sweep in 0..SWEEPS {
            let (lengths, scales) =
                timed(Part::Sizes, || sizes(&neighbors, &elements, coordinates));
            let mut state = Sweep {
                adjacency: &adjacency,
                contributions: &contributions,
                element_chunk,
                elements: &elements,
                epsilon,
                lengths,
                node_chunk,
                node_faces: &node_faces,
                nodes,
                scales,
                sorted: &sorted,
                targets: timed(Part::Targets, || {
                    oracle.targets(&faces, coordinates, face_chunk)
                })?,
                threads,
                tracked: &tracked,
                unknowns,
            };
            let (quality, worst) = timed(Part::Measure, || state.measure(coordinates));
            if sweep > 0 {
                epsilon = schedule(epsilon, quality, previous, worst);
                state.epsilon = epsilon;
            }
            previous = quality;
            let (shift, value, settled) = state.minimize(coordinates)?;
            let stagnant = window.len() == WINDOW
                && window.iter().fold(value, |high, &entry| high.max(entry))
                    - window.iter().fold(value, |low, &entry| low.min(entry))
                    <= value.abs() * STAGNATION;
            if settled || shift < TOLERANCE || stagnant {
                break;
            }
            if window.len() == WINDOW {
                window.pop_front();
            }
            window.push_back(value);
        }
        Ok(())
    }
}

impl<'a> Oracle<'a> {
    fn new(target: &'a Tessellation) -> Self {
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
    fn targets(
        &self,
        faces: &[Vec<usize>],
        coordinates: &Coordinates<3>,
        chunk: usize,
    ) -> Result<Vec<Target>, &'static str> {
        let mut targets = vec![None; faces.len()];
        scope(|scope| {
            targets
                .chunks_mut(chunk)
                .zip(faces.chunks(chunk))
                .for_each(|(targets, faces)| {
                    scope.spawn(move || {
                        targets.iter_mut().zip(faces).for_each(|(target, face)| {
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
                            *target =
                                self.tied_target(&centroid, tolerance)
                                    .map(|(point, normal)| {
                                        let distance = face
                                            .iter()
                                            .map(|&node| {
                                                let deviation =
                                                    (&coordinates[node] - &point) * &normal;
                                                deviation * deviation
                                            })
                                            .fold(Quantity::default(), Quantity::max);
                                        (point, normal, distance)
                                    });
                        })
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
                    .fold(Direction::<3>::const_from([0.0; 3]), |sum, &(other, at)| {
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

impl Sweep<'_> {
    fn measure(&self, coordinates: &Coordinates<3>) -> (Quantity<Length>, Scalar) {
        self.tracked
            .iter()
            .map(|&element| {
                let scale = self.scales[element].value();
                let (corners, nodes) = &self.elements[element];
                (
                    self.scales[element]
                        * energy(corners, nodes, coordinates, scale.powi(3) * self.epsilon),
                    determinant(corners, nodes, coordinates) / scale.powi(3),
                )
            })
            .fold(
                (Quantity::default(), Scalar::INFINITY),
                |(quality, worst), (q, d)| (quality + q, worst.min(d)),
            )
    }
    fn objective(&self, coordinates: &Coordinates<3>) -> Quantity<Length> {
        scope(|scope| {
            self.tracked
                .chunks(self.element_chunk)
                .map(|chunk| {
                    scope.spawn(move || {
                        chunk
                            .iter()
                            .map(|&element| {
                                let (corners, nodes) = &self.elements[element];
                                self.scales[element]
                                    * energy(
                                        corners,
                                        nodes,
                                        coordinates,
                                        self.scales[element].value().powi(3) * self.epsilon,
                                    )
                            })
                            .sum::<Quantity<Length>>()
                    })
                })
                .collect::<Vec<_>>()
                .into_iter()
                .map(|handle| handle.join().unwrap())
                .sum::<Quantity<Length>>()
        }) + scope(|scope| {
            self.nodes
                .chunks(self.node_chunk)
                .map(|chunk| {
                    scope.spawn(move || {
                        chunk
                            .iter()
                            .map(|&node| {
                                BALANCE / self.lengths[node]
                                    * self.node_faces[node]
                                        .iter()
                                        .map(|&face| {
                                            let (point, normal, distance) = &self.targets[face];
                                            let weight = weight(*distance, self.lengths[node]);
                                            let deviation = (&coordinates[node] - point) * normal;
                                            deviation * deviation * weight
                                        })
                                        .sum::<Quantity<Area>>()
                            })
                            .sum::<Quantity<Length>>()
                    })
                })
                .collect::<Vec<_>>()
                .into_iter()
                .map(|handle| handle.join().unwrap())
                .sum::<Quantity<Length>>()
        })
    }
    fn derivative(&self, coordinates: &Coordinates<3>) -> Gradient {
        let mut guard = self.contributions.lock().unwrap();
        let mut rest: &mut [Entry] = guard.as_mut_slice();
        scope(|scope| {
            self.tracked
                .chunks(self.element_chunk)
                .enumerate()
                .for_each(|(block, chunk)| {
                    let first = block * self.element_chunk;
                    let length =
                        self.adjacency.offsets[first + chunk.len()] - self.adjacency.offsets[first];
                    let (mine, tail) = take(&mut rest).split_at_mut(length);
                    rest = tail;
                    scope.spawn(move || {
                        let mut at = 0;
                        chunk.iter().for_each(|&element| {
                            let (corners, nodes) = &self.elements[element];
                            let scale = self.scales[element];
                            nodes
                                .iter()
                                .zip(scatter(
                                    corners,
                                    nodes,
                                    coordinates,
                                    scale.value().powi(3) * self.epsilon,
                                ))
                                .for_each(|(_, contribution)| {
                                    mine[at] = contribution * scale;
                                    at += 1
                                })
                        })
                    });
                })
        });
        let contributions: &[Entry] = &guard;
        let mut gradient = self.empty();
        scope(|scope| {
            gradient
                .as_mut_slice()
                .chunks_mut(self.node_chunk)
                .zip(self.nodes.chunks(self.node_chunk))
                .enumerate()
                .for_each(|(block, (entries, nodes))| {
                    scope.spawn(move || {
                        entries.iter_mut().zip(nodes).enumerate().for_each(
                            |(offset, (entry, &node))| {
                                let index = block * self.node_chunk + offset;
                                self.adjacency.entries[self.adjacency.starts[index]
                                    ..self.adjacency.starts[index + 1]]
                                    .iter()
                                    .for_each(|&at| *entry += &contributions[at]);
                                self.node_faces[node].iter().for_each(|&face| {
                                    let (point, normal, distance) = &self.targets[face];
                                    let weight = weight(*distance, self.lengths[node]);
                                    let deviation = (&coordinates[node] - point) * normal;
                                    let factor =
                                        2.0 * BALANCE / self.lengths[node] * weight * deviation;
                                    *entry += normal * factor.value()
                                })
                            },
                        );
                    });
                });
        });
        gradient
    }
    fn empty(&self) -> Gradient {
        Gradient::from(vec![[0.0; 3]; self.unknowns])
    }
    fn minimize(
        &self,
        coordinates: &mut Coordinates<3>,
    ) -> Result<(Scalar, Quantity<Length>, bool), &'static str> {
        let typical = self
            .nodes
            .iter()
            .map(|&node| self.lengths[node])
            .sum::<Quantity<Length>>()
            / self.nodes.len().max(1) as Scalar;
        let initial: Coordinates<3> = self
            .nodes
            .iter()
            .map(|&node| coordinates[node].clone())
            .collect();
        let residual = CONVERGENCE * initial.norm().value().max(1.0);
        let shared = RefCell::new(coordinates);
        let assign = |x: &Coordinates<3>| {
            timed(Part::Assign, || {
                place(self.sorted, self.threads, &mut shared.borrow_mut(), x)
            })
        };
        let descent = timed(Part::Descent, || {
            solver(typical, residual).descend(
                |x| {
                    assign(x);
                    Ok(timed(Part::Objective, || self.objective(&shared.borrow())))
                },
                |x| {
                    assign(x);
                    Ok(timed(Part::Derivative, || {
                        self.derivative(&shared.borrow())
                    }))
                },
                initial.clone(),
                Vec::new(),
            )
        })
        .map_err(|_| "could not minimize the fit energy")?;
        #[cfg(test)]
        {
            use core::sync::atomic::Ordering::Relaxed;
            profile::STEPS.fetch_add(descent.steps as u64, Relaxed);
            profile::UNKNOWNS.store(self.nodes.len() as u64, Relaxed);
        }
        assign(&descent.solution);
        let shift = self
            .nodes
            .iter()
            .zip(descent.solution.iter().zip(&initial))
            .map(|(&node, (end, start))| ((end - start).norm() / self.lengths[node]).value())
            .fold(0.0, Scalar::max);
        Ok((
            shift,
            descent.value,
            descent.converged && descent.steps == 0,
        ))
    }
}

/// Puts the entries of `x` at their nodes, `sorted` holding each node with the index of its entry
/// in `x`, in the order of the nodes. The nodes are split into ranges, one for each thread, so
/// that no thread writes where another does.
fn place(
    sorted: &[(usize, usize)],
    threads: usize,
    coordinates: &mut Coordinates<3>,
    x: &Coordinates<3>,
) {
    if threads < 2 || sorted.len() < PLACE_GRAIN {
        sorted
            .iter()
            .for_each(|&(node, index)| coordinates[node] = x[index].clone());
        return;
    }
    let chunk = coordinates.len().div_ceil(threads).max(1);
    scope(|scope| {
        coordinates
            .as_mut_slice()
            .chunks_mut(chunk)
            .enumerate()
            .for_each(|(block, entries)| {
                let first = block * chunk;
                let low = sorted.partition_point(|&(node, _)| node < first);
                let high = sorted.partition_point(|&(node, _)| node < first + entries.len());
                let own = &sorted[low..high];
                if !own.is_empty() {
                    scope.spawn(move || {
                        own.iter()
                            .for_each(|&(node, index)| entries[node - first] = x[index].clone())
                    });
                }
            })
    });
}

fn sizes(
    neighbors: &[Vec<usize>],
    elements: &[(&'static CornerTable, Vec<usize>)],
    coordinates: &Coordinates<3>,
) -> (Vec<Quantity<Length>>, Vec<Quantity<Length>>) {
    let lengths: Vec<Quantity<Length>> = (0..coordinates.len())
        .map(|node| {
            neighbors[node]
                .iter()
                .map(|&neighbor| (&coordinates[neighbor] - &coordinates[node]).norm())
                .sum::<Quantity<Length>>()
                / neighbors[node].len().max(1) as Scalar
        })
        .collect();
    let scales = elements
        .iter()
        .map(|(_, element)| {
            element
                .iter()
                .map(|&node| lengths[node])
                .sum::<Quantity<Length>>()
                / element.len() as Scalar
        })
        .collect();
    (lengths, scales)
}

fn schedule(
    epsilon: Scalar,
    quality: Quantity<Length>,
    previous: Quantity<Length>,
    worst: Scalar,
) -> Scalar {
    let sigma = RELAXATION.max((1.0 - quality / previous).value());
    let mu = (1.0 - sigma) * chi(epsilon, worst);
    let epsilon_2021 = if worst < mu {
        2.0 * (mu * (mu - worst)).sqrt()
    } else {
        EPSILON_FLOOR
    };
    let epsilon_1999 = (1.0e-18 + (0.2 * worst).powi(2)).sqrt();
    epsilon_2021.min(epsilon_1999)
}

fn edges(
    corner: usize,
    adjacent: &[usize; 3],
    element: &[usize],
    coordinates: &Coordinates<3>,
) -> EdgeList {
    let origin = &coordinates[element[corner]];
    (0..3)
        .map(|i| (&coordinates[element[adjacent[i]]] - origin).with_unit())
        .collect()
}

fn weight(distance: Quantity<Area>, length: Quantity<Length>) -> Quantity<Dimensionless> {
    1.0 / (distance / (length * length)).max(WEIGHT_FLOOR)
}

fn energy(
    corners: &CornerTable,
    element: &[usize],
    coordinates: &Coordinates<3>,
    epsilon: Scalar,
) -> Scalar {
    corners
        .iter()
        .map(|(corner, adjacent)| {
            regularized(&edges(*corner, adjacent, element, coordinates), epsilon)
        })
        .sum()
}

fn scatter(
    corners: &CornerTable,
    element: &[usize],
    coordinates: &Coordinates<3>,
    epsilon: Scalar,
) -> [Slope; 8] {
    let mut local: [TensorRank1<3, Reference>; 8] = from_fn(|_| TensorRank1::const_from([0.0; 3]));
    corners.iter().for_each(|(corner, adjacent)| {
        let edges = edges(*corner, adjacent, element, coordinates);
        let trace = edges.norm_squared().value();
        let determinant = edges.scalar_triple_product();
        let denominator = chi(epsilon, determinant);
        let alpha = 3.0 * trace.sqrt() / denominator;
        let beta = trace.powf(1.5)
            * 0.5
            * (1.0 + determinant / (epsilon * epsilon + determinant * determinant).sqrt())
            / (denominator * denominator);
        let crosses = [
            edges[1].cross(&edges[2]),
            edges[2].cross(&edges[0]),
            edges[0].cross(&edges[1]),
        ];
        (0..3).for_each(|i| {
            local[*corner] += &crosses[i] * beta - &edges[i] * alpha;
            local[adjacent[i]] += &edges[i] * alpha - &crosses[i] * beta;
        });
    });
    local.map(|entry| entry.with_unit())
}

fn determinant(corners: &CornerTable, element: &[usize], coordinates: &Coordinates<3>) -> Scalar {
    corners
        .iter()
        .map(|(corner, adjacent)| {
            edges(*corner, adjacent, element, coordinates).scalar_triple_product()
        })
        .fold(Scalar::INFINITY, Scalar::min)
}
