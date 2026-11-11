#[cfg(test)]
mod test;

use super::{CornerTable, Oracle, STAGNATION, SWEEPS, Sweep, TOLERANCE, WINDOW, schedule, sizes};
use crate::{
    geometry::mesh::{
        Connectivity, Mesh, Tessellation,
        quality::metrics::{hexahedron, pyramid, tetrahedron, wedge},
    },
    math::{Quantity, Scalar},
    units::Length,
};
use alloc::collections::VecDeque;
use core::mem::replace;
use std::thread::available_parallelism;

impl Mesh<3> {
    /// Fits the nodes to the targets, each face attracted only to the target
    /// of its own patch, so that it cannot snap to a neighbouring surface.
    pub(crate) fn fit_patches(
        &mut self,
        nodes: &[usize],
        patches: &[(&Tessellation, Vec<Vec<usize>>)],
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
        let oracles: Vec<Oracle> = patches
            .iter()
            .map(|&(target, _)| Oracle::new(target))
            .collect();
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
        let (faces, owners): (Vec<Vec<usize>>, Vec<usize>) = patches
            .iter()
            .enumerate()
            .flat_map(|(patch, (_, faces))| faces.iter().map(move |face| (face, patch)))
            .filter(|(face, _)| face.iter().any(|&node| free[node]))
            .map(|(face, patch)| (face.clone(), patch))
            .unzip();
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
        let unknowns = nodes.len();
        let mut epsilon: Scalar = 1.0;
        let mut previous = Quantity::<Length>::new(Scalar::INFINITY);
        let mut window = VecDeque::<Quantity<Length>>::with_capacity(WINDOW);
        for sweep in 0..SWEEPS {
            let (lengths, scales) = sizes(&neighbors, &elements, coordinates);
            let mut state = Sweep {
                element_chunk,
                elements: &elements,
                epsilon,
                lengths,
                node_chunk,
                node_faces: &node_faces,
                nodes,
                scales,
                slot: &slot,
                targets: Oracle::targets(&oracles, &owners, &faces, coordinates, face_chunk)?,
                tracked: &tracked,
                unknowns,
            };
            let (quality, worst) = state.measure(coordinates);
            if sweep > 0 {
                epsilon = schedule(epsilon, quality, previous, worst);
                state.epsilon = epsilon;
            }
            previous = quality;
            let (shift, value, settled) = state.minimize(coordinates);
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
