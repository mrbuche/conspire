mod oracle;
mod patch;
mod sweep;

use self::oracle::Oracle;
use crate::{
    geometry::{
        Coordinate, Direction,
        mesh::{Mesh, Tessellation},
    },
    math::{FxHashSet, Quantity, Scalar},
    units::Area,
};

type Target = (Coordinate<3>, Direction<3>, Quantity<Area>);

type CornerTable = [(usize, [usize; 3])];

const STAGNATION: Scalar = 5.0e-4;
const SWEEPS: usize = 50;
const TOLERANCE: Scalar = 1.0e-3;
const WINDOW: usize = 3;

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
        self.fit_patches(nodes, &[(target, self.exterior_faces())])
    }
}
