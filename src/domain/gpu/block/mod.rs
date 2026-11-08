//! The blocks of a model as the kernels take them.

#[cfg(test)]
mod test;

use crate::domain::block::element::Items;

/// The arrays of a block of elements that the kernels read, with the element last, so that the
/// threads of a warp, which are consecutive elements, read consecutive addresses.
pub struct DeviceLayout {
    /// The number of elements.
    pub items: usize,
    /// The number of nodes of each element.
    pub nodes_per_item: usize,
    /// The node `a` of an element `item`, at `a * items + item`.
    pub connectivity: Vec<u32>,
    /// Where the contributions to each node begin, for the nodes up to the last that an element
    /// has, and where the last ends.
    pub offsets: Vec<u32>,
    /// The contributions to the nodes, in the order of the elements, as `a * items + item` for
    /// the node `a` of an element `item`, which is where its force is in each of the three planes
    /// of components of the forces of the elements.
    pub contributions: Vec<u32>,
}

impl DeviceLayout {
    /// The layout of the elements of a block.
    ///
    /// The contributions to a node are in the order of the elements, so a sum of them is that of
    /// adding the forces of the elements one after the other, as the nodal forces of a block are.
    pub fn new(items: &(impl Items + ?Sized)) -> Self {
        let number_of_items = items.number_of_items();
        let nodes_per_item = if number_of_items == 0 {
            0
        } else {
            items.nodes_of(0).len()
        };
        let index = |value: usize| u32::try_from(value).expect("an index of a device is 32 bits");
        let mut connectivity = vec![0; nodes_per_item * number_of_items];
        (0..number_of_items).for_each(|item| {
            let nodes = items.nodes_of(item);
            assert_eq!(nodes.len(), nodes_per_item, "the elements of a block agree");
            nodes
                .iter()
                .enumerate()
                .for_each(|(a, &node)| connectivity[a * number_of_items + item] = index(node))
        });
        let nodes = (0..number_of_items)
            .flat_map(|item| items.nodes_of(item))
            .max()
            .map_or(0, |&node| node + 1);
        let mut offsets = vec![0_usize; nodes + 1];
        (0..number_of_items)
            .flat_map(|item| items.nodes_of(item))
            .for_each(|&node| offsets[node + 1] += 1);
        (0..nodes).for_each(|node| offsets[node + 1] += offsets[node]);
        let mut next = offsets.clone();
        let mut contributions = vec![0; offsets[nodes]];
        (0..number_of_items).for_each(|item| {
            items
                .nodes_of(item)
                .iter()
                .enumerate()
                .for_each(|(a, &node)| {
                    contributions[next[node]] = index(a * number_of_items + item);
                    next[node] += 1;
                })
        });
        let offsets = offsets.into_iter().map(index).collect();
        Self {
            items: number_of_items,
            nodes_per_item,
            connectivity,
            offsets,
            contributions,
        }
    }
    /// The number of nodes that the offsets are for.
    pub fn number_of_nodes(&self) -> usize {
        self.offsets.len() - 1
    }
    /// The number of values in each of the three planes of components of the forces of the
    /// elements.
    pub fn plane(&self) -> usize {
        self.nodes_per_item * self.items
    }
}

/// A block of a model as the kernels take it: what kernel makes the forces of its elements, the
/// parameters of its model, what its elements share, and the layout of its elements.
pub struct DeviceBlock {
    /// The name of its model in the names of the kernels.
    pub model: &'static str,
    /// The parameters of the model.
    pub parameters: Vec<f64>,
    /// The integration points of each element.
    pub points: usize,
    /// The derivatives of the shape functions at the integration points of the parametric
    /// element, as point, then node, then component, which all of the elements share.
    pub derivatives: Vec<f64>,
    /// The weights of the integration points of the parametric element.
    pub weights: Vec<f64>,
    /// The layout of the elements.
    pub layout: DeviceLayout,
}

/// Something that is made of blocks that a device can run, in the order that their forces are
/// added in.
pub trait DeviceBlocks {
    /// Adds the blocks that it is made of to `blocks`.
    fn device_blocks(&self, blocks: &mut Vec<DeviceBlock>);
}

#[cfg(feature = "fem")]
mod fem {
    use super::{DeviceBlock, DeviceBlocks, DeviceLayout};
    use crate::{
        constitutive::solid::elastic::DeviceModel,
        domain::{Blocks, Model},
        fem::{
            block::Block,
            element::{OnTheFly, linear::LinearElement, solid::elastic::ElasticFiniteElement},
        },
    };

    impl<C, const G: usize, const N: usize, R, H> DeviceBlocks
        for Block<C, LinearElement<G, N>, G, 3, N, N, R, H>
    where
        C: DeviceModel,
        LinearElement<G, N>: ElasticFiniteElement<C, G, 3, N, N>,
    {
        fn device_blocks(&self, blocks: &mut Vec<DeviceBlock>) {
            let (derivatives, weights) = OnTheFly::<G, N>::tables::<LinearElement<G, N>>();
            blocks.push(DeviceBlock {
                model: C::NAME,
                parameters: self.constitutive_model().parameters(),
                points: G,
                derivatives: derivatives.as_flattened().as_flattened().to_vec(),
                weights: weights.to_vec(),
                layout: DeviceLayout::new(self),
            })
        }
    }

    impl<B1, B2> DeviceBlocks for Blocks<B1, B2>
    where
        B1: DeviceBlocks,
        B2: DeviceBlocks,
    {
        fn device_blocks(&self, blocks: &mut Vec<DeviceBlock>) {
            self.0.device_blocks(blocks);
            self.1.device_blocks(blocks)
        }
    }

    impl<B> DeviceBlocks for Model<B, 3>
    where
        B: DeviceBlocks,
    {
        fn device_blocks(&self, blocks: &mut Vec<DeviceBlock>) {
            self.blocks().device_blocks(blocks)
        }
    }
}
