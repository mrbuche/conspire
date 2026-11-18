#[cfg(test)]
mod test;

use crate::{
    geometry::{
        Coordinate, Coordinates,
        mesh::{Connectivities, Connectivity, Mesh, from::kuhn, retain::subset},
        solid::{Partition, SolidOracle},
    },
    math::{Quantity, Tensor, TensorVec},
};
use alloc::collections::BTreeSet;
use core::array::from_fn;

fn lattice_size(
    lower: &Coordinate<3>,
    upper: &Coordinate<3>,
    nel: [usize; 3],
) -> Result<Coordinate<3>, &'static str> {
    if nel.contains(&0) {
        return Err("the lattice needs at least one cell along each axis");
    }
    let size = Coordinate::from(from_fn(|a| (upper[a] - lower[a]) / nel[a] as f64));
    if (0..3).any(|a| size[a] <= Quantity::default()) {
        return Err("the upper corner must be above the lower one along each axis");
    }
    Ok(size)
}

fn lattice_cells(nel: [usize; 3]) -> impl Iterator<Item = [usize; 3]> {
    (0..nel[2])
        .flat_map(move |k| (0..nel[1]).flat_map(move |j| (0..nel[0]).map(move |i| [i, j, k])))
}

fn lattice_point(lower: &Coordinate<3>, size: &Coordinate<3>, at: [f64; 3]) -> Coordinate<3> {
    from_fn(|a| lower[a] + size[a] * at[a]).into()
}

impl Mesh<3> {
    /// Hexahedra on a uniform lattice between `lower` and `upper`, one block per volume of the
    /// partition, labeled by its index.
    ///
    /// A cell belongs to the volume owning its center and is dropped where no volume does.
    /// Neighbouring blocks share the nodes of the faces between them, so the mesh conforms to
    /// every interface as a staircase.
    pub fn from_partition<O: SolidOracle>(
        partition: &Partition<O>,
        lower: &Coordinate<3>,
        upper: &Coordinate<3>,
        nel: [usize; 3],
    ) -> Result<Self, &'static str> {
        let size = lattice_size(lower, upper, nel)?;
        Ok(Self::from_lattice_cells(
            lattice_cells(nel).filter_map(|cell| {
                let center = lattice_point(lower, &size, cell.map(|index| index as f64 + 0.5));
                partition.owner(&center).map(|volume| (cell, volume))
            }),
            nel,
            &size,
            lower,
        ))
    }
    /// Tetrahedra on a uniform lattice between `lower` and `upper`, six to each cell as
    /// the lattice tetrahedra have them, one block per volume of the partition, labeled by its
    /// index.
    ///
    /// A tetrahedron belongs to the volume owning its centroid and is dropped where no volume
    /// does, so the staircase between blocks is finer than that of [`Mesh::from_partition`], its
    /// steps being tetrahedra rather than cells. Neighbouring blocks share the nodes of the faces
    /// between them.
    pub fn from_partition_tets<O: SolidOracle>(
        partition: &Partition<O>,
        lower: &Coordinate<3>,
        upper: &Coordinate<3>,
        nel: [usize; 3],
    ) -> Result<Self, &'static str> {
        let size = lattice_size(lower, upper, nel)?;
        let tets = kuhn(&from_fn(|corner| corner));
        Ok(Self::from_lattice_tet_materials(
            lattice_cells(nel).filter_map(|cell| {
                let owners = tets.map(|tet| {
                    let centroid = from_fn(|a| {
                        cell[a] as f64
                            + tet.iter().map(|&corner| (corner >> a) & 1).sum::<usize>() as f64
                                / 4.0
                    });
                    partition.owner(&lattice_point(lower, &size, centroid))
                });
                owners.iter().any(Option::is_some).then_some((cell, owners))
            }),
            nel,
            &size,
            lower,
        ))
    }
    /// The mesh with its elements in one block per volume of the partition, labeled by its index.
    ///
    /// An element belongs to the volume owning its centroid, the mean of its nodes, and is dropped
    /// where no volume does, as are the nodes only it had. A volume owning no element has no
    /// block. Like [`Mesh::from_partition`] and [`Mesh::from_partition_tets`], the interface
    /// between two blocks is a staircase, but the mesh may be any of a tree's, as long as it is
    /// of a single block of one kind of primitive element.
    pub fn label_by_partition<O: SolidOracle>(
        &self,
        partition: &Partition<O>,
    ) -> Result<Self, &'static str> {
        let mut blocks = self.iter();
        let (Some(block), None) = (blocks.next(), blocks.next()) else {
            return Err("only a mesh of a single block can be labeled");
        };
        if matches!(
            block,
            Connectivity::Polyhedral(_) | Connectivity::Polygonal(_)
        ) {
            return Err("polytopal elements have no centroid to be labeled by");
        }
        let coordinates = self.coordinates();
        let owners: Vec<Option<usize>> = block
            .iter()
            .map(|element| {
                let centroid = from_fn(|axis| {
                    element
                        .iter()
                        .map(|&node| coordinates[node][axis].value())
                        .sum::<f64>()
                        / element.len() as f64
                })
                .into();
                partition.owner(&centroid)
            })
            .collect();
        let labels: Vec<usize> = owners
            .iter()
            .flatten()
            .copied()
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        if labels.is_empty() {
            return Err("no volume owns any element");
        }
        let mut remap = vec![usize::MAX; coordinates.len()];
        let mut kept_coordinates = Coordinates::new();
        let mut id = |node: usize| {
            if remap[node] == usize::MAX {
                remap[node] = kept_coordinates.len();
                kept_coordinates.push(coordinates[node].clone())
            }
            remap[node]
        };
        let connectivities: Vec<Connectivity> = labels
            .iter()
            .map(|&label| {
                let kept: Vec<&[usize]> = block
                    .iter()
                    .zip(&owners)
                    .filter(|&(_, &owner)| owner == Some(label))
                    .map(|(element, _)| element)
                    .collect();
                subset(block, &kept, &mut id)
            })
            .collect();
        Ok((
            Connectivities::from((connectivities, labels)),
            kept_coordinates,
        )
            .into())
    }
}
