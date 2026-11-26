#[cfg(test)]
mod test;

use super::StabilizedElement;
use crate::{
    domain::block::device::DeviceKind,
    fem::element::{
        DeviceElement,
        mass::{ElementNodalLumpedMasses, IntegrationDensities, LumpedMassFiniteElement},
    },
    math::Quantity,
    units::{RateSquared, Stress},
};

impl DeviceElement<1, 8> for StabilizedElement<8> {
    const KIND: DeviceKind = DeviceKind::Stabilized;
    const VALUES: usize = StabilizedElement::<8>::VALUES;
    fn shared() -> (Vec<f64>, Vec<f64>) {
        (Vec::new(), Vec::new())
    }
    fn values(&self) -> Vec<f64> {
        StabilizedElement::values(self)
    }
    fn lumped_masses(
        &self,
        densities: &IntegrationDensities<1>,
    ) -> Option<ElementNodalLumpedMasses<8>> {
        Some(self.nodal_lumped_masses(densities))
    }
    fn added_squared_frequency(
        &self,
        shear_modulus: Quantity<Stress>,
        nodal_masses: &ElementNodalLumpedMasses<8>,
    ) -> Quantity<RateSquared> {
        self.hourglass_squared_frequency_bound(shear_modulus, nodal_masses)
    }
}
