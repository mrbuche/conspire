#[cfg(test)]
mod test;

use super::{CornerTable, Target};
use crate::{
    geometry::{
        Coordinates,
        mesh::quality::metrics::{chi, regularized},
    },
    math::{
        CrossProduct, Quantity, Reference, Scalar, Tensor, TensorRank1, TensorRank1List,
        TensorRank1Vec,
        optimize::{Armijo, LimitedMemoryBFGS, Tolerances},
    },
    units::{Area, Dimensionless, Length, ReciprocalLength},
};
use core::{array::from_fn, cell::RefCell};
use std::thread::scope;

type EdgeList = TensorRank1List<3, Reference, 3>;
type Slope = TensorRank1<3, Reference, ReciprocalLength>;
type Gradient = TensorRank1Vec<3, Reference, Dimensionless>;

const BACKTRACKS: usize = 32;
const BALANCE: Scalar = 2.5e3;
const CONVERGENCE: Scalar = 1.0e-5;
const EPSILON_FLOOR: Scalar = 1.0e-12;
const HISTORY: usize = 8;
const ITERATIONS: usize = 100;
const RELAXATION: Scalar = 0.1;
const WEIGHT_FLOOR: Quantity = Dimensionless::of(0.3);

fn solver(typical: Quantity<Length>, residual: Scalar) -> LimitedMemoryBFGS<Armijo, Length> {
    LimitedMemoryBFGS {
        abs_tol: Tolerances {
            constraint: 0.0,
            residual,
        },
        history: HISTORY,
        line_search: Armijo {
            control: 1.0e-4,
            cut_back: 0.5,
            max_steps: BACKTRACKS,
        },
        max_steps: ITERATIONS,
        rel_tol: None,
        typical,
    }
}

pub(super) struct Elements {
    pub(super) corners: Vec<&'static CornerTable>,
    pub(super) nodes: Vec<usize>,
    pub(super) offsets: Vec<usize>,
}

impl Elements {
    pub(super) fn len(&self) -> usize {
        self.corners.len()
    }
    fn get(&self, index: usize) -> (&'static CornerTable, &[usize]) {
        (
            self.corners[index],
            &self.nodes[self.offsets[index]..self.offsets[index + 1]],
        )
    }
}

pub(super) struct Sweep<'a> {
    pub(super) element_chunk: usize,
    pub(super) elements: &'a Elements,
    pub(super) epsilon: Scalar,
    pub(super) lengths: Vec<Quantity<Length>>,
    pub(super) node_chunk: usize,
    pub(super) node_faces: &'a [Vec<usize>],
    pub(super) nodes: &'a [usize],
    pub(super) scales: Vec<Quantity<Length>>,
    pub(super) slot: &'a [Option<usize>],
    pub(super) targets: Vec<Target>,
    pub(super) tracked: &'a [usize],
    pub(super) unknowns: usize,
}

impl Sweep<'_> {
    pub(super) fn measure(&self, coordinates: &Coordinates<3>) -> (Quantity<Length>, Scalar) {
        self.tracked
            .iter()
            .map(|&element| {
                let scale = self.scales[element].value();
                let (corners, nodes) = self.elements.get(element);
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
                                let (corners, nodes) = self.elements.get(element);
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
        let partials = scope(|scope| {
            self.tracked
                .chunks(self.element_chunk)
                .map(|chunk| {
                    scope.spawn(move || {
                        let mut partial = self.empty();
                        chunk.iter().for_each(|&element| {
                            let (corners, nodes) = self.elements.get(element);
                            let local = scatter(
                                corners,
                                nodes,
                                coordinates,
                                self.scales[element].value().powi(3) * self.epsilon,
                            );
                            nodes.iter().zip(local).for_each(|(&node, contribution)| {
                                if let Some(index) = self.slot[node] {
                                    partial[index] += contribution * self.scales[element]
                                }
                            })
                        });
                        partial
                    })
                })
                .collect::<Vec<_>>()
                .into_iter()
                .map(|handle| handle.join().unwrap())
                .collect::<Vec<_>>()
        });
        let mut gradient = self.empty();
        scope(|scope| {
            let partials = &partials;
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
                                partials
                                    .iter()
                                    .for_each(|partial| *entry += &partial[index]);
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
        (0..self.unknowns)
            .map(|_| TensorRank1::const_from([0.0; 3]))
            .collect()
    }
    pub(super) fn minimize(
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
            let mut coordinates = shared.borrow_mut();
            self.nodes
                .iter()
                .zip(x.iter())
                .for_each(|(&node, entry)| coordinates[node] = entry.clone())
        };
        let descent = solver(typical, residual)
            .descend(
                |x| {
                    assign(x);
                    Ok(self.objective(&shared.borrow()))
                },
                |x| {
                    assign(x);
                    Ok(self.derivative(&shared.borrow()))
                },
                initial.clone(),
                Vec::new(),
            )
            .map_err(|_| "could not minimize the fit energy")?;
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

pub(super) fn sizes(
    neighbors: &[Vec<usize>],
    elements: &Elements,
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
    let scales = (0..elements.len())
        .map(|index| {
            let (_, element) = elements.get(index);
            element
                .iter()
                .map(|&node| lengths[node])
                .sum::<Quantity<Length>>()
                / element.len() as Scalar
        })
        .collect();
    (lengths, scales)
}

pub(super) fn schedule(
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

pub(super) fn energy(
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

pub(super) fn scatter(
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
