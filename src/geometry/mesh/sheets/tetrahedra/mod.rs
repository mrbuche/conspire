#[cfg(test)]
mod test;

use crate::{
    geometry::mesh::{Connectivities, Connectivity, Mesh},
    math::{FxHashMap, FxHashSet},
};
use alloc::collections::BTreeMap;
use core::array::from_fn;

const FACES: [[usize; 3]; 4] = [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]];

const EDGES: [[usize; 2]; 6] = [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]];

impl Mesh<3> {
    pub(super) fn resolve_tetrahedral_pinches(&mut self) -> Result<usize, &'static str> {
        let labels = self.block_labels().ok_or("every block needs a label")?;
        let mut tets = Vec::<[usize; 4]>::new();
        let mut owners = Vec::<usize>::new();
        for (block, &label) in self.iter().zip(labels) {
            let Connectivity::Tetrahedral(elements) = block else {
                return Err("pinches are only resolved between tetrahedra");
            };
            for element in elements.iter() {
                tets.push(from_fn(|node| element[node]));
                owners.push(label);
            }
        }
        let adjacent = adjacent(&tets);
        let rings = rings(&tets, &adjacent);
        let stars = stars(&tets, self.number_of_nodes());
        let mut relabeled = 0;
        let mut passes = 0;
        loop {
            let before = relabeled;
            for ring in &rings {
                relabeled += pinch_ring(ring, &mut owners);
            }
            for star in &stars {
                relabeled += pinch_star(star, &adjacent, &mut owners);
            }
            if relabeled == before {
                break;
            }
            passes += 1;
            if passes > tets.len() {
                return Err("the pinches could not be resolved");
            }
        }
        if relabeled > 0 {
            let mut grouped = BTreeMap::<usize, Vec<[usize; 4]>>::new();
            tets.into_iter()
                .zip(owners)
                .for_each(|(tet, label)| grouped.entry(label).or_default().push(tet));
            let (blocks, labels): (Vec<Connectivity>, Vec<usize>) = grouped
                .into_iter()
                .map(|(label, elements)| (Connectivity::Tetrahedral(elements.into()), label))
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

fn adjacent(tets: &[[usize; 4]]) -> Vec<Vec<usize>> {
    let mut owners = FxHashMap::<[usize; 3], Vec<usize>>::default();
    for (index, tet) in tets.iter().enumerate() {
        for face in FACES {
            let mut key = face.map(|node| tet[node]);
            key.sort_unstable();
            owners.entry(key).or_default().push(index);
        }
    }
    let mut adjacent = vec![Vec::new(); tets.len()];
    for sharing in owners.into_values() {
        if let [first, second] = sharing[..] {
            adjacent[first].push(second);
            adjacent[second].push(first);
        }
    }
    adjacent
}

fn rings(tets: &[[usize; 4]], adjacent: &[Vec<usize>]) -> Vec<Vec<usize>> {
    let mut around = FxHashMap::<[usize; 2], Vec<usize>>::default();
    for (index, tet) in tets.iter().enumerate() {
        for edge in EDGES {
            let [a, b] = edge.map(|node| tet[node]);
            around.entry([a.min(b), a.max(b)]).or_default().push(index);
        }
    }
    let mut around: Vec<_> = around.into_iter().collect();
    around.sort_unstable();
    around
        .into_iter()
        .filter_map(|(_, members)| {
            let inside: FxHashSet<usize> = members.iter().copied().collect();
            let beside = |tet: usize| -> Vec<usize> {
                adjacent[tet]
                    .iter()
                    .copied()
                    .filter(|other| inside.contains(other))
                    .collect()
            };
            let start = members[0];
            let mut ring = vec![start];
            let mut previous = start;
            let mut current = *beside(start).first()?;
            while current != start {
                let next = beside(current);
                if next.len() != 2 || ring.len() > members.len() {
                    return None;
                }
                ring.push(current);
                let following = if next[0] == previous {
                    next[1]
                } else {
                    next[0]
                };
                previous = current;
                current = following;
            }
            (ring.len() == members.len()).then_some(ring)
        })
        .collect()
}

fn stars(tets: &[[usize; 4]], nodes: usize) -> Vec<Vec<usize>> {
    let mut stars = vec![Vec::new(); nodes];
    for (index, tet) in tets.iter().enumerate() {
        tet.iter().for_each(|&node| stars[node].push(index));
    }
    stars
}

fn pinch_ring(ring: &[usize], owners: &mut [usize]) -> usize {
    let Some(start) = (0..ring.len())
        .find(|&at| owners[ring[at]] != owners[ring[(at + ring.len() - 1) % ring.len()]])
    else {
        return 0;
    };
    let mut runs = Vec::<(usize, Vec<usize>)>::new();
    for at in 0..ring.len() {
        let tet = ring[(start + at) % ring.len()];
        match runs.last_mut() {
            Some((label, members)) if *label == owners[tet] => members.push(tet),
            _ => runs.push((owners[tet], vec![tet])),
        }
    }
    if runs.len() < 4 {
        return 0;
    }
    let lowest = (0..runs.len())
        .min_by_key(|&run| (runs[run].0, *runs[run].1.iter().min().unwrap_or(&0)))
        .unwrap_or(0);
    let label = runs[(lowest + runs.len() - 1) % runs.len()]
        .0
        .max(runs[(lowest + 1) % runs.len()].0);
    runs[lowest].1.iter().for_each(|&tet| owners[tet] = label);
    runs[lowest].1.len()
}

fn pinch_star(star: &[usize], adjacent: &[Vec<usize>], owners: &mut [usize]) -> usize {
    let inside: FxHashSet<usize> = star.iter().copied().collect();
    let mut seen = FxHashSet::<usize>::default();
    let mut islands = Vec::<(usize, Vec<usize>)>::new();
    for &start in star {
        if !seen.insert(start) {
            continue;
        }
        let label = owners[start];
        let mut island = vec![start];
        let mut pending = vec![start];
        while let Some(tet) = pending.pop() {
            for &other in &adjacent[tet] {
                if inside.contains(&other) && owners[other] == label && seen.insert(other) {
                    island.push(other);
                    pending.push(other);
                }
            }
        }
        islands.push((label, island));
    }
    let repeated = islands
        .iter()
        .any(|(label, _)| islands.iter().filter(|(other, _)| other == label).count() > 1);
    if !repeated {
        return 0;
    }
    let (label, members) = islands
        .iter()
        .filter(|(label, _)| islands.iter().filter(|(other, _)| other == label).count() > 1)
        .min_by_key(|(label, members)| (members.len(), *label, members.iter().min().copied()))
        .expect("a repeated label has islands");
    let mut shared = BTreeMap::<usize, usize>::new();
    members
        .iter()
        .flat_map(|&tet| adjacent[tet].iter().copied())
        .filter(|other| inside.contains(other) && owners[*other] != *label)
        .for_each(|other| *shared.entry(owners[other]).or_default() += 1);
    let Some((&around, _)) = shared.iter().max_by_key(|&(&label, &faces)| (faces, label)) else {
        return 0;
    };
    members.iter().for_each(|&tet| owners[tet] = around);
    members.len()
}
