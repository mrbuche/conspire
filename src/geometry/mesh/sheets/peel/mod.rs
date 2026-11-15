#[cfg(test)]
mod test;

use crate::{
    geometry::mesh::{
        Connectivities, Connectivity, Mesh, PrimitiveConnectivity, buffer::prism_tets,
    },
    math::{FxHashMap, FxHashSet, Tensor, TensorVec},
};
use alloc::collections::BTreeMap;
use core::{array::from_fn, cmp::Reverse};

/// The middle faces of the sheets between a pair of labels, the higher label first.
type Sheet = ((usize, usize), Vec<Vec<usize>>);

/// A mesh peeled along its sheets.
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
    /// gets two layers of elements, one in each block, between the copies of that block and the
    /// middle ones, a hexahedron or, for a triangle, a prism of three tetrahedra, see
    /// [`prism_tets`]. The mesh is as it was in shape, as the new elements have no volume yet and
    /// the new nodes are at the old ones, but every element that touches a sheet has a face on it,
    /// and when the middle nodes move to a surface the layers open around it.
    ///
    /// The hexahedra must not have an edge pinched, see [`Mesh::resolve_pinches`]. It fails where
    /// the blocks are not all hexahedral or all tetrahedral, the labels of the blocks are missing
    /// or not distinct, a face is shared by more than two elements, or a sheet reaches the
    /// boundary of the mesh.
    pub(super) fn double_peel(&self) -> Result<DoublePeeled, &'static str> {
        match self.iter().next() {
            Some(Connectivity::Hexahedral(_)) => self.peel_sheets::<4, 8>(
                |connectivity| matches!(connectivity, Connectivity::Hexahedral(_)),
                |bottom, top| {
                    vec![from_fn(|node| {
                        if node < 4 {
                            bottom[node]
                        } else {
                            top[node - 4]
                        }
                    })]
                },
                Connectivity::Hexahedral,
            ),
            Some(Connectivity::Tetrahedral(_)) => self.peel_sheets::<3, 4>(
                |connectivity| matches!(connectivity, Connectivity::Tetrahedral(_)),
                |bottom, top| prism_tets(*bottom, *top).to_vec(),
                Connectivity::Tetrahedral,
            ),
            _ => Err("only hexahedra or tetrahedra can be peeled"),
        }
    }
    /// Peels the sheets of elements of `F`-node faces and `N` nodes, filling between the copies
    /// of a face with the elements `layer` gives for the faces below and above it.
    fn peel_sheets<const F: usize, const N: usize>(
        &self,
        of_kind: fn(&Connectivity) -> bool,
        layer: impl Fn(&[usize; F], &[usize; F]) -> Vec<[usize; N]>,
        variant: fn(PrimitiveConnectivity<3, N>) -> Connectivity,
    ) -> Result<DoublePeeled, &'static str> {
        let labels = self.block_labels().ok_or("every block needs a label")?;
        if labels.iter().collect::<FxHashSet<_>>().len() != labels.len() {
            return Err("the blocks need distinct labels");
        }
        let mut owners = FxHashMap::<[usize; F], Vec<(usize, [usize; F])>>::default();
        let mut blocks_of = vec![Vec::<usize>::new(); self.number_of_nodes()];
        let mut blocks = Vec::<Vec<[usize; N]>>::new();
        for (block, connectivity) in self.iter().enumerate() {
            if !of_kind(connectivity) {
                return Err("every block must be of the one kind peeled");
            }
            let mut elements = Vec::new();
            for element in connectivity.iter() {
                elements.push(from_fn(|node| element[node]));
                element.iter().for_each(|&node| {
                    if !blocks_of[node].contains(&block) {
                        blocks_of[node].push(block)
                    }
                });
                for face in connectivity.element_faces(element) {
                    let face: [usize; F] = from_fn(|node| face[node]);
                    let mut key = face;
                    key.sort_unstable();
                    owners.entry(key).or_default().push((block, face));
                }
            }
            blocks.push(elements);
        }
        blocks_of
            .iter_mut()
            .for_each(|of| of.sort_unstable_by_key(|&block| Reverse(labels[block])));
        let mut exterior = FxHashSet::<usize>::default();
        let mut sheets = Vec::<(usize, usize, [usize; F])>::new();
        for entries in owners.into_values() {
            match entries[..] {
                [(_, face)] => exterior.extend(face),
                [(a, first), (b, second)] if a != b => sheets.push(if labels[a] > labels[b] {
                    (a, b, first)
                } else {
                    (b, a, second)
                }),
                [_, _] => {}
                _ => return Err("a face is shared by more than two elements"),
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
        blocks.iter_mut().enumerate().for_each(|(block, elements)| {
            elements
                .iter_mut()
                .flatten()
                .for_each(|node| *node = copy_of(*node, block))
        });
        let mut middles = BTreeMap::<(usize, usize), Vec<Vec<usize>>>::new();
        for &(higher, lower, face) in &sheets {
            let low = face.map(|node| copy_of(node, higher));
            let mid = face.map(|node| middle[&shared(node, pair(higher, lower))]);
            let high = face.map(|node| copy_of(node, lower));
            blocks[higher].extend(layer(&low, &mid));
            blocks[lower].extend(layer(&mid, &high));
            middles
                .entry(pair(higher, lower))
                .or_default()
                .push(mid.to_vec());
        }
        let free = (0..coordinates.len())
            .filter(|node| !exterior.contains(node))
            .collect();
        let connectivities: Vec<Connectivity> = blocks
            .into_iter()
            .map(|elements| variant(elements.into()))
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
