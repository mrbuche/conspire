#[cfg(test)]
mod test;

use crate::math::Quantity;
use crate::{
    fem::element::{FiniteElement, IntegrationData},
    math::{Reference, Tensor, TensorRank1List},
    units::{ReciprocalLength, Volume},
};

/// The integration data of one element of a block, made as the integration points are gone
/// through, from the reference coordinates of its nodes and the derivatives of the shape
/// functions at the integration points of the parametric element, which every element of the
/// block shares.
///
/// With $`\mathbf{J}`$ the sum over the nodes of the outer products of the coordinates and the
/// derivatives, the gradient vector of a node is the inverse transpose of $`\mathbf{J}`$ times
/// its derivative, and the weight is the determinant of $`\mathbf{J}`$ times the weight of the
/// point in the parametric element, as an element makes them when it is made. They are made again
/// at every use, which is more arithmetic and less memory traffic than reading them, so it is
/// faster where the memory is the limit, as it is on many devices. They are the same as those of
/// the element to rounding, and not to the bit, since the inverse is by the cofactors.
///
/// The derivatives are `derivatives[point][node][component]`, and the weights `weights[point]`,
/// which [`tables`](Self::tables) makes for a type of element.
pub struct OnTheFly<'a, const G: usize, const N: usize> {
    nodes: [[f64; 3]; N],
    derivatives: &'a [[[f64; 3]; N]; G],
    weights: &'a [f64; G],
}

impl<'a, const G: usize, const N: usize> OnTheFly<'a, G, N> {
    /// The data of the element with the given reference coordinates of its nodes.
    pub const fn new(
        nodes: [[f64; 3]; N],
        derivatives: &'a [[[f64; 3]; N]; G],
        weights: &'a [f64; G],
    ) -> Self {
        Self {
            nodes,
            derivatives,
            weights,
        }
    }
    /// The derivatives of the shape functions of an element at its integration points, and the
    /// weights of the points, in the parametric element.
    pub fn tables<F>() -> ([[[f64; 3]; N]; G], [f64; G])
    where
        F: FiniteElement<G, 3, N, N>,
    {
        let mut derivatives = [[[0.0; 3]; N]; G];
        F::shape_functions_gradients_at_integration_points()
            .iter()
            .zip(derivatives.iter_mut())
            .for_each(|(gradients, derivatives)| {
                gradients
                    .iter()
                    .zip(derivatives.iter_mut())
                    .for_each(|(gradient, derivatives)| {
                        gradient
                            .iter()
                            .zip(derivatives.iter_mut())
                            .for_each(|(value, derivative)| *derivative = value.value())
                    })
            });
        let mut weights = [0.0; G];
        F::parametric_weights()
            .iter()
            .zip(weights.iter_mut())
            .for_each(|(value, weight)| *weight = *value);
        (derivatives, weights)
    }
    /// The cofactors of $`\mathbf{J}`$ at an integration point, which are its determinant times
    /// its inverse transpose, and its determinant.
    fn cofactors(&self, point: usize) -> ([[f64; 3]; 3], f64) {
        let mut j = [[0.0; 3]; 3];
        self.nodes
            .iter()
            .zip(self.derivatives[point].iter())
            .for_each(|(coordinates, derivatives)| {
                (0..3).for_each(|i| (0..3).for_each(|k| j[i][k] += coordinates[i] * derivatives[k]))
            });
        let mut cofactors = [[0.0; 3]; 3];
        (0..3).for_each(|i| {
            let (i1, i2) = ((i + 1) % 3, (i + 2) % 3);
            (0..3).for_each(|k| {
                let (k1, k2) = ((k + 1) % 3, (k + 2) % 3);
                cofactors[i][k] = j[i1][k1] * j[i2][k2] - j[i1][k2] * j[i2][k1];
            })
        });
        let determinant =
            j[0][0] * cofactors[0][0] + j[0][1] * cofactors[0][1] + j[0][2] * cofactors[0][2];
        (cofactors, determinant)
    }
}

impl<const G: usize, const N: usize> IntegrationData<G, N> for OnTheFly<'_, G, N> {
    fn gradient_vectors_at(
        &self,
        point: usize,
    ) -> TensorRank1List<3, Reference, N, ReciprocalLength> {
        let (cofactors, determinant) = self.cofactors(point);
        let reciprocal = 1.0 / determinant;
        let mut gradient_vectors = [[0.0; 3]; N];
        gradient_vectors
            .iter_mut()
            .zip(self.derivatives[point].iter())
            .for_each(|(gradient_vector, derivatives)| {
                gradient_vector
                    .iter_mut()
                    .enumerate()
                    .for_each(|(i, value)| {
                        *value = (cofactors[i][0] * reciprocal) * derivatives[0]
                            + (cofactors[i][1] * reciprocal) * derivatives[1]
                            + (cofactors[i][2] * reciprocal) * derivatives[2]
                    })
            });
        gradient_vectors.into()
    }
    fn integration_weight_at(&self, point: usize) -> Quantity<Volume> {
        let (_, determinant) = self.cofactors(point);
        Volume::cubic_meters(determinant * self.weights[point])
    }
}
