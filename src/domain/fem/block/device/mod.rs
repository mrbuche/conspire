use super::Block;
use crate::{
    constitutive::FromParameters,
    domain::block::device::{DeviceBlock, DeviceBlocks, DeviceLayout},
    fem::element::{FiniteElement, IntegrationData, Recomputed},
    math::Tensor,
};

impl<C, E, const G: usize, const N: usize, R, H> DeviceBlocks for Block<C, E, G, 3, N, N, R, H>
where
    C: FromParameters,
    E: FiniteElement<G, 3, N, N> + IntegrationData<G, N>,
{
    fn device_blocks(&self, blocks: &mut Vec<DeviceBlock>) {
        let (derivatives, weights) = Recomputed::<3, G, N>::tables::<E>();
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
