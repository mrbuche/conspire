#[cfg(test)]
mod test;

use super::IntegrationData;
use crate::{
    fem::element::{FiniteElement, StandardGradientOperators},
    math::{Quantity, Reference, ScalarList, TensorRank1List},
    units::{Length, ReciprocalLength, Volume},
};

/// The integration data of one element of a block, recomputed as the integration points are gone
/// through, from the reference coordinates of its nodes and the derivatives of the shape
/// functions at the integration points of the parametric element, which every element of the
/// block shares.
///
/// With $`\mathbf{J}`$ the sum over the nodes of the outer products of the coordinates and the
/// derivatives, the gradient vector of a node is the inverse transpose of $`\mathbf{J}`$ times
/// its derivative, and the weight is the determinant of $`\mathbf{J}`$ times the weight of the
/// point in the parametric element. An element stores them when it is made, by making them here
/// at every point, so they are the same to the bit. Made again at every use, they are more
/// arithmetic and less memory traffic than reading them, so faster where the memory is the limit,
/// as it is on many devices.
///
/// The derivatives are `derivatives[point][node]`, and the weights `weights[point]`, which
/// [`tables`](Self::tables) makes for a type of element.
pub struct Recomputed<'a, const D: usize, const G: usize, const N: usize> {
    nodes: TensorRank1List<D, Reference, N, Length>,
    derivatives: &'a StandardGradientOperators<D, N, G>,
    weights: &'a ScalarList<G>,
}

impl<'a, const D: usize, const G: usize, const N: usize> Recomputed<'a, D, G, N> {
    /// The data of the element with the given reference coordinates of its nodes.
    pub const fn new(
        nodes: TensorRank1List<D, Reference, N, Length>,
        derivatives: &'a StandardGradientOperators<D, N, G>,
        weights: &'a ScalarList<G>,
    ) -> Self {
        Self {
            nodes,
            derivatives,
            weights,
        }
    }
    /// The shape function gradients and weights at the integration points.
    pub fn tables<F>() -> (StandardGradientOperators<D, N, G>, ScalarList<G>)
    where
        F: FiniteElement<G, D, N, N>,
    {
        (
            F::shape_functions_gradients_at_integration_points(),
            F::parametric_weights(),
        )
    }
    /// The gradient vectors of the nodes at an integration point.
    pub fn gradient_vectors(
        &self,
        point: usize,
    ) -> TensorRank1List<D, Reference, N, ReciprocalLength> {
        let derivatives = &self.derivatives[point];
        (&self.nodes * derivatives).inverse_transpose() * derivatives
    }
    /// The weight of an integration point.
    pub fn integration_weight(&self, point: usize) -> Quantity<Volume> {
        Quantity::new((&self.nodes * &self.derivatives[point]).determinant() * self.weights[point])
    }
}

impl<const G: usize, const N: usize> IntegrationData<G, N> for Recomputed<'_, 3, G, N> {
    fn gradient_vectors_at(
        &self,
        point: usize,
    ) -> TensorRank1List<3, Reference, N, ReciprocalLength> {
        self.gradient_vectors(point)
    }
    fn integration_weight_at(&self, point: usize) -> Quantity<Volume> {
        self.integration_weight(point)
    }
}
