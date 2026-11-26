#[cfg(test)]
mod test;

use super::{Element, FiniteElement, Recomputed};
use crate::{
    domain::block::device::DeviceKind,
    fem::element::mass::{ElementNodalLumpedMasses, IntegrationDensities, LumpedMassFiniteElement},
    math::{Quantity, Tensor},
    units::{RateSquared, Stress},
};

/// An element that a device can take: what its block shares, and what each element has of its
/// own, which the kernel of its [kind](DeviceKind) reads.
pub trait DeviceElement<const G: usize, const N: usize>: FiniteElement<G, 3, N, N> {
    /// The kind of the element.
    const KIND: DeviceKind;
    /// The number of values that an element has of its own.
    const VALUES: usize;
    /// What the elements of a block share: the derivatives of the shape functions at the
    /// integration points of the parametric element, as point, then node, then component, and the
    /// weights of the points. An element that shares nothing has neither.
    fn shared() -> (Vec<f64>, Vec<f64>);
    /// The values of the element, which are [`VALUES`](Self::VALUES) of them.
    fn values(&self) -> Vec<f64>;
    /// The lumped masses of the nodes of the element, if its time step can be checked.
    fn lumped_masses(
        &self,
        densities: &IntegrationDensities<G>,
    ) -> Option<ElementNodalLumpedMasses<N>>;
    /// What the element adds to the bound on its squared angular frequency beyond what its
    /// integration points give, which does not change with its coordinates.
    fn added_squared_frequency(
        &self,
        _shear_modulus: Quantity<Stress>,
        _nodal_masses: &ElementNodalLumpedMasses<N>,
    ) -> Quantity<RateSquared> {
        Quantity::default()
    }
}

/// What the elements of a block share, for elements that integrate at all of their points.
fn tables<E, const G: usize, const N: usize>() -> (Vec<f64>, Vec<f64>)
where
    E: FiniteElement<G, 3, N, N>,
{
    let (derivatives, weights) = Recomputed::tables::<E>();
    (
        derivatives
            .iter()
            .flat_map(|point| point.iter())
            .flat_map(|node| node.iter())
            .map(|derivative| derivative.value())
            .collect(),
        weights.iter().copied().collect(),
    )
}

impl<const G: usize, const N: usize> DeviceElement<G, N> for Element<3, G, N, 1>
where
    Self: LumpedMassFiniteElement<G, 3, N, N>,
{
    const KIND: DeviceKind = DeviceKind::Full;
    const VALUES: usize = 0;
    fn shared() -> (Vec<f64>, Vec<f64>) {
        tables::<Self, G, N>()
    }
    fn values(&self) -> Vec<f64> {
        Vec::new()
    }
    fn lumped_masses(
        &self,
        densities: &IntegrationDensities<G>,
    ) -> Option<ElementNodalLumpedMasses<N>> {
        Some(self.nodal_lumped_masses(densities))
    }
}

impl<const G: usize, const N: usize> DeviceElement<G, N> for Element<3, G, N, 2>
where
    Self: FiniteElement<G, 3, N, N>,
{
    const KIND: DeviceKind = DeviceKind::Full;
    const VALUES: usize = 0;
    fn shared() -> (Vec<f64>, Vec<f64>) {
        tables::<Self, G, N>()
    }
    fn values(&self) -> Vec<f64> {
        Vec::new()
    }
    fn lumped_masses(
        &self,
        _densities: &IntegrationDensities<G>,
    ) -> Option<ElementNodalLumpedMasses<N>> {
        None
    }
}
