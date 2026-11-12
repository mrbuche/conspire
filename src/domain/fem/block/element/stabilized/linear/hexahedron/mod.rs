#[cfg(test)]
mod test;

use crate::{
    fem::block::element::{
        ElementNodalReferenceCoordinates, FiniteElement, IntegrationWeights, ParametricCoordinate,
        ParametricCoordinates, ParametricReference, ShapeFunctions, ShapeFunctionsGradients,
        linear::Hexahedron as FullHexahedron,
        stabilized::{HourglassBaseVectors, M, StabilizedElement, stabilized_from},
    },
    math::{ScalarList, TensorRank1},
    units::Volume,
};

const G: usize = 1;
const N: usize = 8;
const P: usize = N;
const K: usize = 4;

const BASE_VECTORS: HourglassBaseVectors<K, N> = HourglassBaseVectors::const_from([
    TensorRank1::const_from([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0]),
    TensorRank1::const_from([1.0, 1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0]),
    TensorRank1::const_from([1.0, -1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0]),
    TensorRank1::const_from([-1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0]),
]);

pub type Hexahedron = StabilizedElement<N>;

impl FiniteElement<G, M, N, P> for Hexahedron {
    fn integration_points() -> ParametricCoordinates<G, M> {
        [[0.0, 0.0, 0.0]].into()
    }
    fn integration_weights(&self) -> &IntegrationWeights<G, Volume> {
        &self.integration_weights
    }
    fn parametric_reference() -> ParametricReference<M, N> {
        FullHexahedron::parametric_reference()
    }
    fn parametric_weights() -> ScalarList<G> {
        [8.0].into()
    }
    fn shape_functions(parametric_coordinate: ParametricCoordinate<M>) -> ShapeFunctions<N> {
        FullHexahedron::shape_functions(parametric_coordinate)
    }
    fn shape_functions_gradients(
        parametric_coordinate: ParametricCoordinate<M>,
    ) -> ShapeFunctionsGradients<M, N> {
        FullHexahedron::shape_functions_gradients(parametric_coordinate)
    }
}

impl From<ElementNodalReferenceCoordinates<N>> for Hexahedron {
    fn from(reference_nodal_coordinates: ElementNodalReferenceCoordinates<N>) -> Self {
        stabilized_from(
            &FullHexahedron::from(reference_nodal_coordinates.clone()),
            &reference_nodal_coordinates,
            &BASE_VECTORS,
        )
    }
}
