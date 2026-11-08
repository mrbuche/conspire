use super::DeviceLayout;
use crate::domain::block::element::Items;

#[derive(Debug)]
struct Quads(Vec<[usize; 4]>);

impl Items for Quads {
    fn number_of_items(&self) -> usize {
        self.0.len()
    }
    fn nodes_of(&self, item: usize) -> &[usize] {
        &self.0[item]
    }
}

fn strip() -> Quads {
    Quads(vec![
        [0, 1, 4, 3],
        [1, 2, 5, 4],
        [3, 4, 7, 6],
        [4, 5, 8, 7],
        [8, 9, 12, 11],
    ])
}

#[test]
fn the_nodes_of_the_elements_are_in_the_layout_with_the_element_last() {
    let items = strip();
    let layout = DeviceLayout::new(&items);
    assert_eq!(layout.items, 5);
    assert_eq!(layout.nodes_per_item, 4);
    assert_eq!(layout.plane(), 20);
    assert_eq!(layout.connectivity.len(), 20);
    items.0.iter().enumerate().for_each(|(item, nodes)| {
        nodes
            .iter()
            .enumerate()
            .for_each(|(a, &node)| assert_eq!(layout.connectivity[a * 5 + item], node as u32))
    });
}

#[test]
fn the_contributions_to_a_node_are_those_of_the_elements_that_have_it_in_their_order() {
    let items = strip();
    let layout = DeviceLayout::new(&items);
    assert_eq!(layout.number_of_nodes(), 13);
    assert_eq!(layout.offsets.len(), 14);
    assert_eq!(layout.offsets[0], 0);
    assert_eq!(
        *layout.offsets.last().unwrap() as usize,
        layout.contributions.len()
    );
    assert_eq!(layout.contributions.len(), 20);
    for node in 0..layout.number_of_nodes() {
        let expected = items
            .0
            .iter()
            .enumerate()
            .flat_map(|(item, nodes)| {
                nodes
                    .iter()
                    .enumerate()
                    .filter(move |&(_, &n)| n == node)
                    .map(move |(a, _)| (a * 5 + item) as u32)
            })
            .collect::<Vec<_>>();
        let (first, last) = (
            layout.offsets[node] as usize,
            layout.offsets[node + 1] as usize,
        );
        assert_eq!(layout.contributions[first..last], expected, "node {node}");
    }
}

#[test]
fn a_node_that_no_element_has_has_no_contributions() {
    let layout = DeviceLayout::new(&strip());
    assert_eq!(layout.offsets[10], layout.offsets[11]);
    assert_eq!(layout.offsets[0], layout.offsets[1] - 1);
}

#[test]
fn no_elements_are_no_layout() {
    let layout = DeviceLayout::new(&Quads(vec![]));
    assert_eq!(layout.items, 0);
    assert_eq!(layout.nodes_per_item, 0);
    assert_eq!(layout.number_of_nodes(), 0);
    assert!(layout.connectivity.is_empty() && layout.contributions.is_empty());
}

fn gather(layout: &DeviceLayout, forces: &[f64], nodal_forces: &mut [[f64; 3]]) {
    let plane = layout.plane();
    for (node, nodal_force) in nodal_forces
        .iter_mut()
        .enumerate()
        .take(layout.number_of_nodes())
    {
        let (first, last) = (
            layout.offsets[node] as usize,
            layout.offsets[node + 1] as usize,
        );
        let mut sum = *nodal_force;
        for k in first..last {
            let contribution = layout.contributions[k] as usize;
            sum.iter_mut()
                .enumerate()
                .for_each(|(i, value)| *value += forces[i * plane + contribution]);
        }
        *nodal_force = sum;
    }
}

#[test]
fn the_sum_of_the_gather_is_the_sum_of_the_elements_one_after_the_other_to_the_bit() {
    let items = strip();
    let layout = DeviceLayout::new(&items);
    let value = |item: usize, a: usize, i: usize| {
        let k = (item * 4 + a) * 3 + i;
        [1.0e16, 1.0, -1.0e16, 0.1][k % 4] + 0.001 * k as f64
    };
    let mut forces = vec![0.0; 3 * layout.plane()];
    for item in 0..5 {
        for a in 0..4 {
            for i in 0..3 {
                forces[i * layout.plane() + a * 5 + item] = value(item, a, i);
            }
        }
    }
    let mut expected = vec![[0.0; 3]; 13];
    for (item, nodes) in items.0.iter().enumerate() {
        for (a, &node) in nodes.iter().enumerate() {
            expected[node]
                .iter_mut()
                .enumerate()
                .for_each(|(i, force)| *force += value(item, a, i));
        }
    }
    let mut found = vec![[0.0; 3]; 13];
    gather(&layout, &forces, &mut found);
    for (found, expected) in found.iter().zip(&expected) {
        for (found, expected) in found.iter().zip(expected) {
            assert_eq!(found.to_bits(), expected.to_bits());
        }
    }
    let mut reversed = vec![[0.0; 3]; 13];
    for (item, nodes) in items.0.iter().enumerate().rev() {
        for (a, &node) in nodes.iter().enumerate() {
            reversed[node]
                .iter_mut()
                .enumerate()
                .for_each(|(i, force)| *force += value(item, a, i));
        }
    }
    assert!(
        reversed
            .iter()
            .flatten()
            .zip(expected.iter().flatten())
            .any(|(reversed, expected)| reversed.to_bits() != expected.to_bits())
    );
}
