#[cfg(test)]
mod test;

use super::{Element, FiniteElement, Recomputed};
use crate::{domain::block::device::DeviceKind, math::Tensor};

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
}

impl<const G: usize, const N: usize, const O: usize> DeviceElement<G, N> for Element<3, G, N, O>
where
    Self: FiniteElement<G, 3, N, N>,
{
    const KIND: DeviceKind = DeviceKind::Full;
    const VALUES: usize = 0;
    fn shared() -> (Vec<f64>, Vec<f64>) {
        let (derivatives, weights) = Recomputed::tables::<Self>();
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
    fn values(&self) -> Vec<f64> {
        Vec::new()
    }
}
