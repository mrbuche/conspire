use super::Block;
use crate::{
    constitutive::{FromParameters, solid::elastic::Elastic},
    domain::block::device::{DeviceBlock, DeviceBlocks, DeviceLayout},
    fem::element::{Recomputed, linear::LinearElement, solid::elastic::ElasticFiniteElement},
    math::Tensor,
};

impl<C, const G: usize, const N: usize, R, H> DeviceBlocks
    for Block<C, LinearElement<G, N>, G, 3, N, N, R, H>
where
    C: Elastic + FromParameters,
    LinearElement<G, N>: ElasticFiniteElement<C, G, 3, N, N>,
{
    fn device_blocks(&self, blocks: &mut Vec<DeviceBlock>) {
        let (derivatives, weights) = Recomputed::<3, G, N>::tables::<LinearElement<G, N>>();
        blocks.push(DeviceBlock {
            model: C::NAME,
            parameters: self.constitutive_model().parameters(),
            points: G,
            derivatives: derivatives
                .iter()
                .flat_map(|point| point.iter())
                .flat_map(|node| node.iter())
                .map(|derivative| derivative.value())
                .collect(),
            weights: weights.iter().copied().collect(),
            layout: DeviceLayout::new(self),
        })
    }
}
