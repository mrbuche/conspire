#[cfg(test)]
mod test;

use super::peel::DoublePeeled;
use crate::geometry::mesh::{Interfaces, Mesh, Tessellation};

impl Mesh<3> {
    /// Deforms the mesh so that the faces between its blocks lie on the surfaces between the
    /// materials of the interfaces.
    ///
    /// The label of each block is the index of its material among the interfaces, as when the
    /// mesh comes from [`Mesh::from_partition`] of the solids the interfaces are of. Pinched
    /// edges are relabeled away first, see [`Mesh::resolve_pinches`], then the mesh is peeled
    /// twice along every sheet between blocks, and the middle of each is fitted to the surface
    /// between its two materials, and to that one alone, so that it cannot snap to another.
    /// The nodes on the boundary of the mesh stay where they are.
    ///
    /// The node, side and element sets are dropped. It fails where the mesh cannot be peeled, see
    /// [`Mesh::resolve_pinches`], or where two materials of a sheet have no surface between them.
    pub fn conform(&mut self, interfaces: &Interfaces) -> Result<(), &'static str> {
        self.resolve_pinches()?;
        let DoublePeeled {
            mut mesh,
            sheets,
            free,
        } = self.double_peel()?;
        let targets: Vec<Tessellation> = sheets
            .iter()
            .map(|&((higher, lower), _)| interfaces.between(higher, lower))
            .collect::<Result<_, _>>()?;
        let patches: Vec<(&Tessellation, Vec<Vec<usize>>)> = targets
            .iter()
            .zip(sheets)
            .map(|(target, (_, faces))| (target, faces.iter().map(|face| face.to_vec()).collect()))
            .collect();
        mesh.fit_patches(&free, &patches)?;
        *self = mesh;
        Ok(())
    }
}
