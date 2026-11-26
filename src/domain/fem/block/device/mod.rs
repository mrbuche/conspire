use super::{Block, Densities, density::NoDensity};
use crate::{
    constitutive::{FromParameters, solid::Solid},
    domain::block::device::{DeviceBlock, DeviceBlocks, DeviceLayout},
    fem::element::{DeviceElement, mass::IntegrationDensities},
    math::Tensor,
};

/// The densities of a block as a device takes them, if the block has any.
pub trait DeviceDensities<const G: usize> {
    /// The densities at the integration points of an element.
    fn device_densities(&self, element: usize) -> Option<IntegrationDensities<G>>;
}

impl<R, const G: usize> DeviceDensities<G> for R
where
    R: Densities<G>,
{
    fn device_densities(&self, element: usize) -> Option<IntegrationDensities<G>> {
        Some(self.at(element))
    }
}

impl<const G: usize> DeviceDensities<G> for NoDensity {
    fn device_densities(&self, _element: usize) -> Option<IntegrationDensities<G>> {
        None
    }
}

impl<C, E, const G: usize, const N: usize, R, H> DeviceBlocks for Block<C, E, G, 3, N, N, R, H>
where
    C: FromParameters + Solid,
    E: DeviceElement<G, N>,
    R: DeviceDensities<G>,
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
        let lumped = self
            .elements()
            .iter()
            .enumerate()
            .map(|(item, element)| {
                let densities = self.density().device_densities(item)?;
                element.lumped_masses(&densities)
            })
            .collect::<Option<Vec<_>>>()
            .unwrap_or_default();
        let mut masses = vec![0.0; N * lumped.len()];
        for (item, nodal_masses) in lumped.iter().enumerate() {
            for (a, mass) in nodal_masses.iter().enumerate() {
                masses[a * items + item] = mass.value();
            }
        }
        let shear_modulus = self.constitutive_model().shear_modulus();
        let added_squared_frequencies = self
            .elements()
            .iter()
            .zip(&lumped)
            .map(|(element, nodal_masses)| {
                element
                    .added_squared_frequency(shear_modulus, nodal_masses)
                    .value()
            })
            .collect();
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
            masses,
            added_squared_frequencies,
        })
    }
}
