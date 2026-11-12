pub mod linear;

use crate::{
    fem::block::element::{
        Element, ElementNodalReferenceCoordinates, FiniteElement, GradientVectors,
        IntegrationWeights,
    },
    math::{Projection, Quantity, Tensor, TensorRank1, TensorRank1List},
    units::{Length, Volume},
};
use core::{
    array::from_fn,
    fmt::{self, Debug, Formatter},
    iter::once,
};

const M: usize = 3;

/// Hourglass base vectors, one row per hourglass mode.
pub type HourglassBaseVectors<const K: usize, const N: usize> = TensorRank1List<N, Projection, K>;

/// Hourglass matrix, already scaled by the geometric factor `B:B/V`, so that
/// the hourglass force is the material factor times its product with the
/// nodal coordinates.
pub type HourglassMatrix<const N: usize> = [[Quantity<Length>; N]; N];

/// A one-point (uniform gradient) element whose deformation gradient comes
/// from the exact mean gradient, to be stabilized against hourglass modes.
#[derive(Clone)]
pub struct StabilizedElement<const N: usize> {
    gradient_vectors: GradientVectors<3, 1, N>,
    integration_weights: IntegrationWeights<1, Volume>,
    hourglass_matrix: HourglassMatrix<N>,
    nodal_volumes: IntegrationWeights<N, Volume>,
}

impl<const N: usize> StabilizedElement<N> {
    pub fn gradient_vectors(&self) -> &GradientVectors<3, 1, N> {
        &self.gradient_vectors
    }
    pub fn hourglass_matrix(&self) -> &HourglassMatrix<N> {
        &self.hourglass_matrix
    }
    pub fn nodal_volumes(&self) -> &IntegrationWeights<N, Volume> {
        &self.nodal_volumes
    }
}

impl<const N: usize> Debug for StabilizedElement<N> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let element = match N {
            8 => "StabilizedHexahedron",
            _ => panic!(),
        };
        write!(f, "{element} {{ integration points: 1, nodes: {N} }}")
    }
}

/// Builds the stabilized element from the fully integrated one.
///
/// The mean gradient and the volume come from the quadrature of the full
/// element, which is exact for them, and the hourglass vectors are the base
/// vectors with their projection onto the linear fields removed.
fn stabilized_from<const G: usize, const N: usize, const O: usize, const K: usize>(
    full: &Element<3, G, N, O>,
    reference_nodal_coordinates: &ElementNodalReferenceCoordinates<N>,
    base_vectors: &HourglassBaseVectors<K, N>,
) -> StabilizedElement<N>
where
    Element<3, G, N, O>: FiniteElement<G, 3, N, N>,
{
    let weights = full.integration_weights();
    let volume = weights.iter().sum();
    let mean_gradients = (0..N)
        .map(|a| {
            full.gradient_vectors()
                .iter()
                .zip(weights.iter())
                .map(|(gradient_vectors, weight)| &gradient_vectors[a] * weight)
                .sum::<TensorRank1<_, _, _>>()
                / volume
        })
        .collect::<TensorRank1List<_, _, _, _>>();
    let hourglass_vectors = base_vectors
        .iter()
        .map(|base_vector| {
            let projection = base_vector
                .iter()
                .zip(reference_nodal_coordinates.iter())
                .map(|(base_vector_a, coordinate)| coordinate * base_vector_a)
                .sum::<TensorRank1<_, _, _>>();
            from_fn(|a| (base_vector[a] - &projection * &mean_gradients[a]) / 8.0)
        })
        .collect::<Vec<[_; N]>>();
    let scale = mean_gradients
        .iter()
        .map(|mean_gradient| &(mean_gradient * volume) * mean_gradient)
        .sum::<Quantity<_>>();
    let hourglass_matrix = from_fn(|a| {
        from_fn(|b| {
            scale
                * hourglass_vectors
                    .iter()
                    .map(|hourglass_vector| hourglass_vector[a] * hourglass_vector[b])
                    .sum::<Quantity>()
        })
    });
    let shape_functions = Element::shape_functions_at_integration_points();
    let nodal_volumes = (0..N)
        .map(|a| {
            weights
                .iter()
                .zip(&shape_functions)
                .map(|(weight, shape_functions)| weight * shape_functions[a])
                .sum()
        })
        .collect();
    StabilizedElement {
        gradient_vectors: once(mean_gradients).collect(),
        integration_weights: [volume].into(),
        hourglass_matrix,
        nodal_volumes,
    }
}
