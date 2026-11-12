mod conform;
mod peel;
#[cfg(test)]
mod test;

use crate::{
    geometry::mesh::{Connectivities, Connectivity, Mesh},
    math::FxHashMap,
};
use alloc::collections::BTreeMap;
use core::array::from_fn;

const EDGES: [[usize; 2]; 12] = [
    [0, 1],
    [1, 2],
    [2, 3],
    [3, 0],
    [4, 5],
    [5, 6],
    [6, 7],
    [7, 4],
    [0, 4],
    [1, 5],
    [2, 6],
    [3, 7],
];

impl Mesh<3> {
    /// Relabels hexahedra until no edge is pinched, and gives the number of relabelings.
    ///
    /// An edge is pinched where four hexahedra surround it and each differs in label from the two
    /// beside it, so that the surfaces between the blocks cross along the edge and no sheet
    /// between two blocks can be kept apart from the others there. Where an edge is pinched the
    /// hexahedron around it with the lowest label, or the lowest index among those, takes the
    /// higher label of the two beside it, so the later block wins as where solids overlap. That is
    /// not the label that best keeps the volumes of the blocks, only a repeatable one, and as each
    /// relabeling raises a label it cannot undo another forever.
    ///
    /// The nodes, the other hexahedra and the order of the blocks by label are kept, but the
    /// node, side and element sets are dropped when any hexahedron is relabeled. It fails where a
    /// block is not hexahedral or has no label.
    pub fn resolve_pinches(&mut self) -> Result<usize, &'static str> {
        let labels = self.block_labels().ok_or("every block needs a label")?;
        let mut hexahedra = Vec::<[usize; 8]>::new();
        let mut owners = Vec::<usize>::new();
        for (block, &label) in self.iter().zip(labels) {
            let Connectivity::Hexahedral(elements) = block else {
                return Err("pinches are only resolved between hexahedra");
            };
            for element in elements.iter() {
                hexahedra.push(from_fn(|node| element[node]));
                owners.push(label);
            }
        }
        let rings = rings(&hexahedra);
        let mut relabeled = 0;
        loop {
            let before = relabeled;
            for &ring in &rings {
                let labels = ring.map(|hexahedron| owners[hexahedron]);
                if (0..4).all(|at| labels[at] != labels[(at + 1) % 4]) {
                    let lowest = (1..4).fold(0, |lowest, at| {
                        if (labels[at], ring[at]) < (labels[lowest], ring[lowest]) {
                            at
                        } else {
                            lowest
                        }
                    });
                    owners[ring[lowest]] = labels[(lowest + 1) % 4].max(labels[(lowest + 3) % 4]);
                    relabeled += 1;
                }
            }
            if relabeled == before {
                break;
            }
        }
        if relabeled > 0 {
            let mut grouped = BTreeMap::<usize, Vec<[usize; 8]>>::new();
            hexahedra
                .into_iter()
                .zip(owners)
                .for_each(|(hexahedron, label)| grouped.entry(label).or_default().push(hexahedron));
            let (blocks, labels): (Vec<Connectivity>, Vec<usize>) = grouped
                .into_iter()
                .map(|(label, elements)| (Connectivity::Hexahedral(elements.into()), label))
                .unzip();
            *self = (
                Connectivities::from((blocks, labels)),
                self.coordinates().clone(),
            )
                .into();
        }
        Ok(relabeled)
    }
}

fn rings(hexahedra: &[[usize; 8]]) -> Vec<[usize; 4]> {
    let mut around = FxHashMap::<[usize; 2], Vec<usize>>::default();
    for (index, hexahedron) in hexahedra.iter().enumerate() {
        for edge in EDGES {
            let [a, b] = edge.map(|node| hexahedron[node]);
            around.entry([a.min(b), a.max(b)]).or_default().push(index);
        }
    }
    let beside = |a: usize, b: usize| {
        hexahedra[a]
            .iter()
            .filter(|node| hexahedra[b].contains(node))
            .count()
            >= 4
    };
    let mut rings: Vec<[usize; 4]> = around
        .into_values()
        .filter(|around| around.len() == 4)
        .filter_map(|around| {
            let first = around[0];
            let neighbors: Vec<usize> = around[1..]
                .iter()
                .copied()
                .filter(|&other| beside(first, other))
                .collect();
            let [second, last] = <[usize; 2]>::try_from(neighbors).ok()?;
            let opposite = *around
                .iter()
                .find(|&&other| ![first, second, last].contains(&other))?;
            (beside(second, opposite) && beside(opposite, last)).then_some([
                first,
                second.min(last),
                opposite,
                second.max(last),
            ])
        })
        .collect();
    rings.sort_unstable();
    rings
}
