#[cfg(test)]
mod test;

use crate::{
    geometry::mesh::{Connectivities, Connectivity, Mesh},
    math::{FxHashMap, FxHashSet, Tensor, TensorVec},
};
use alloc::collections::BTreeMap;
use core::{array::from_fn, cmp::Reverse};

/// The middle faces of the sheets between a pair of labels, the higher label first.
type Sheet = ((usize, usize), Vec<[usize; 4]>);

/// A mesh peeled along its sheets.
#[cfg_attr(
    not(test),
    expect(dead_code, reason = "used once the mesh is conformed")
)]
pub(super) struct DoublePeeled {
    pub(super) mesh: Mesh<3>,
    /// The sheets between each pair of labels.
    pub(super) sheets: Vec<Sheet>,
    /// The nodes that are not on the boundary of the mesh.
    pub(super) free: Vec<usize>,
}

impl Mesh<3> {
    /// Peels the mesh along its sheets, the faces between two blocks, into two layers each.
    ///
    /// Each block gets its own copy of the nodes on the sheets, except the block of the highest
    /// label, which keeps the originals. Each sheet gets one node for each of its own nodes in
    /// between the copies, shared by every sheet that meets at that node, and each face of a sheet
    /// gets two hexahedra, one in each block, between the copies of that block and the middle
    /// ones. The mesh is as it was in shape, as the new hexahedra have no volume yet and the new
    /// nodes are at the old ones, but every hexahedron that touches a sheet has a face on it, and
    /// when the middle nodes move to a surface the layers open around it.
    ///
    /// The hexahedra must not have an edge pinched, see [`Mesh::resolve_pinches`]. It fails where
    /// a block is not hexahedral, the labels of the blocks are missing or not distinct, a face is
    /// shared by more than two hexahedra, or a sheet reaches the boundary of the mesh.
    #[cfg_attr(
        not(test),
        expect(dead_code, reason = "used once the mesh is conformed")
    )]
    pub(super) fn double_peel(&self) -> Result<DoublePeeled, &'static str> {
        let labels = self.block_labels().ok_or("every block needs a label")?;
        if labels.iter().collect::<FxHashSet<_>>().len() != labels.len() {
            return Err("the blocks need distinct labels");
        }
        let mut owners = FxHashMap::<[usize; 4], Vec<(usize, [usize; 4])>>::default();
        let mut blocks_of = vec![Vec::<usize>::new(); self.number_of_nodes()];
        let mut blocks = Vec::<Vec<[usize; 8]>>::new();
        for (block, connectivity) in self.iter().enumerate() {
            if !matches!(connectivity, Connectivity::Hexahedral(_)) {
                return Err("only hexahedra can be peeled");
            }
            let mut hexahedra = Vec::new();
            for element in connectivity.iter() {
                hexahedra.push(from_fn(|node| element[node]));
                element.iter().for_each(|&node| {
                    if !blocks_of[node].contains(&block) {
                        blocks_of[node].push(block)
                    }
                });
                for face in connectivity.element_faces(element) {
                    let face: [usize; 4] = from_fn(|node| face[node]);
                    let mut key = face;
                    key.sort_unstable();
                    owners.entry(key).or_default().push((block, face));
                }
            }
            blocks.push(hexahedra);
        }
        blocks_of
            .iter_mut()
            .for_each(|of| of.sort_unstable_by_key(|&block| Reverse(labels[block])));
        let mut exterior = FxHashSet::<usize>::default();
        let mut sheets = Vec::<(usize, usize, [usize; 4])>::new();
        for entries in owners.into_values() {
            match entries[..] {
                [(_, face)] => exterior.extend(face),
                [(a, first), (b, second)] if a != b => sheets.push(if labels[a] > labels[b] {
                    (a, b, first)
                } else {
                    (b, a, second)
                }),
                [_, _] => {}
                _ => return Err("a face is shared by more than two hexahedra"),
            }
        }
        sheets.sort_unstable();
        if sheets
            .iter()
            .any(|(_, _, face)| face.iter().any(|node| exterior.contains(node)))
        {
            return Err("a sheet reaches the boundary of the mesh");
        }
        let pair = |higher: usize, lower: usize| (labels[higher], labels[lower]);
        let mut patches_of = vec![Vec::<(usize, usize)>::new(); self.number_of_nodes()];
        for &(higher, lower, face) in &sheets {
            for node in face {
                if !patches_of[node].contains(&pair(higher, lower)) {
                    patches_of[node].push(pair(higher, lower))
                }
            }
        }
        let shared = |node: usize, pair: (usize, usize)| {
            (node, (patches_of[node].len() < 2).then_some(pair))
        };
        let mut coordinates = self.coordinates().clone();
        let mut duplicate = |node: usize| {
            let point = coordinates[node].clone();
            coordinates.push(point);
            coordinates.len() - 1
        };
        let mut on_sheets: Vec<usize> = sheets
            .iter()
            .flat_map(|(_, _, face)| face.iter().copied())
            .collect();
        on_sheets.sort_unstable();
        on_sheets.dedup();
        let mut copy = FxHashMap::<(usize, usize), usize>::default();
        for &node in &on_sheets {
            for &block in blocks_of[node].iter().skip(1) {
                copy.insert((node, block), duplicate(node));
            }
        }
        let mut middle = FxHashMap::<(usize, Option<(usize, usize)>), usize>::default();
        for &(higher, lower, face) in &sheets {
            for node in face {
                middle
                    .entry(shared(node, pair(higher, lower)))
                    .or_insert_with(|| duplicate(node));
            }
        }
        let copy_of = |node: usize, block: usize| copy.get(&(node, block)).copied().unwrap_or(node);
        blocks
            .iter_mut()
            .enumerate()
            .for_each(|(block, hexahedra)| {
                hexahedra
                    .iter_mut()
                    .flatten()
                    .for_each(|node| *node = copy_of(*node, block))
            });
        let mut middles = BTreeMap::<(usize, usize), Vec<[usize; 4]>>::new();
        for &(higher, lower, face) in &sheets {
            let low = face.map(|node| copy_of(node, higher));
            let mid = face.map(|node| middle[&shared(node, pair(higher, lower))]);
            let high = face.map(|node| copy_of(node, lower));
            let join = |bottom: [usize; 4], top: [usize; 4]| -> [usize; 8] {
                from_fn(|node| {
                    if node < 4 {
                        bottom[node]
                    } else {
                        top[node - 4]
                    }
                })
            };
            blocks[higher].push(join(low, mid));
            blocks[lower].push(join(mid, high));
            middles.entry(pair(higher, lower)).or_default().push(mid);
        }
        let free = (0..coordinates.len())
            .filter(|node| !exterior.contains(node))
            .collect();
        let connectivities: Vec<Connectivity> = blocks
            .into_iter()
            .map(|hexahedra| Connectivity::Hexahedral(hexahedra.into()))
            .collect();
        Ok(DoublePeeled {
            mesh: (
                Connectivities::from((connectivities, labels.to_vec())),
                coordinates,
            )
                .into(),
            sheets: middles.into_iter().collect(),
            free,
        })
    }
}
