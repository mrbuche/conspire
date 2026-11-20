use super::Block;
use crate::{
    constitutive::FromParameters,
    domain::block::device::{DeviceBlock, DeviceBlocks, DeviceLayout},
    fem::element::DeviceElement,
};

impl<C, E, const G: usize, const N: usize, R, H> DeviceBlocks for Block<C, E, G, 3, N, N, R, H>
where
    C: FromParameters,
    E: DeviceElement<G, N>,
{
    fn device_blocks(&self, blocks: &mut Vec<DeviceBlock>) {
        let (derivatives, weights) = E::shared();
        let items = self.elements().len();
        let mut values = vec![0.0; E::VALUES * items];
        for (item, element) in self.elements().iter().enumerate() {
            let element_values = element.values();
            assert_eq!(element_values.len(), E::VALUES, "an element has its values");
            for (k, value) in element_values.into_iter().enumerate() {
                values[k * items + item] = value;
            }
        }
        blocks.push(DeviceBlock {
            kind: E::KIND,
            model: C::NAME,
            parameters: self.constitutive_model().parameters(),
            points: G,
            derivatives,
            weights,
            values_per_item: E::VALUES,
            values,
            layout: DeviceLayout::new(self),
        })
    }
}
