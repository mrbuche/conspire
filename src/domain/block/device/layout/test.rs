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
