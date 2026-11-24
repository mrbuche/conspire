#[cfg(test)]
mod test;

mod recomputed;

pub use recomputed::Recomputed;

use super::{Element, GradientVectors};
use crate::{
    math::{Quantity, Reference, TensorRank1List},
    units::{ReciprocalLength, Volume},
};

impl<const D: usize, const G: usize, const N: usize, const O: usize> Element<D, G, N, O> {
    pub(crate) fn gradient_vectors(&self) -> &GradientVectors<D, G, N> {
        &self.gradient_vectors
    }
}

/// The gradient vectors and the weight of each integration point, stored or recomputed.
pub trait IntegrationData<const G: usize, const N: usize> {
    fn gradient_vectors_at(
        &self,
        point: usize,
    ) -> TensorRank1List<3, Reference, N, ReciprocalLength>;
    fn integration_weight_at(&self, point: usize) -> Quantity<Volume>;
}

impl<const G: usize, const N: usize, const O: usize> IntegrationData<G, N> for Element<3, G, N, O> {
    fn gradient_vectors_at(
        &self,
        point: usize,
    ) -> TensorRank1List<3, Reference, N, ReciprocalLength> {
        self.gradient_vectors[point].clone()
    }
    fn integration_weight_at(&self, point: usize) -> Quantity<Volume> {
        self.integration_weights[point]
    }
}
