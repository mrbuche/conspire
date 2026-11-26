#[cfg(test)]
mod test;

use crate::{
    constitutive::solid::Solid,
    domain::block::element::time_scale::courant_friedrichs_lewy::CourantFriedrichsLewyElement,
    domain::block::element::time_scale::courant_friedrichs_lewy::point_stiffness::{
        GradientGram, PointStiffness,
    },
    fem::element::{
        Element, ElementNodalCoordinates, FiniteElementError, IntegrationData,
        mass::ElementNodalLumpedMasses, stabilized::StabilizedElement,
    },
    math::{Quantity, Tensor, TensorRank2},
    mechanics::DeformationGradient,
    units::RateSquared,
};

/// The sum over the integration points of what each contributes to the bound, with the
/// integration data of the element from `data`.
pub fn squared_frequency_bound_from<C, S, D, const G: usize, const N: usize>(
    data: &D,
    constitutive_model: &C,
    stiffness: &S,
    nodal_coordinates: &ElementNodalCoordinates<N>,
    nodal_masses: &ElementNodalLumpedMasses<N>,
) -> Result<Quantity<RateSquared>, FiniteElementError>
where
    S: PointStiffness<C>,
    D: IntegrationData<G, N>,
{
    let mut squared_frequency = Quantity::default();
    for point in 0..G {
        let gradient_vectors = data.gradient_vectors_at(point);
        let integration_weight = data.integration_weight_at(point);
        let deformation_gradient = nodal_coordinates
            .iter()
            .zip(&gradient_vectors)
            .map(|(nodal_coordinate, gradient_vector)| {
                DeformationGradient::from((nodal_coordinate, gradient_vector))
            })
            .sum();
        let gradient_gram: GradientGram = gradient_vectors
            .iter()
            .zip(nodal_masses.iter())
            .map(|(gradient_vector, mass)| {
                TensorRank2::from((gradient_vector * integration_weight, gradient_vector)) / *mass
            })
            .sum();
        squared_frequency += stiffness
            .squared_frequency(constitutive_model, &deformation_gradient, &gradient_gram)
            .map_err(|error| FiniteElementError::new(Some(point), error))?;
    }
    Ok(squared_frequency)
}

impl<C, S, const G: usize, const N: usize, const O: usize> CourantFriedrichsLewyElement<C, S>
    for Element<3, G, N, O>
where
    S: PointStiffness<C>,
{
    type Masses = ElementNodalLumpedMasses<N>;
    type Error = FiniteElementError;
    fn squared_frequency_bound(
        &self,
        constitutive_model: &C,
        stiffness: &S,
        nodal_coordinates: &ElementNodalCoordinates<N>,
        nodal_masses: &ElementNodalLumpedMasses<N>,
    ) -> Result<Quantity<RateSquared>, FiniteElementError> {
        squared_frequency_bound_from(
            self,
            constitutive_model,
            stiffness,
            nodal_coordinates,
            nodal_masses,
        )
    }
}

impl<C, S, const N: usize> CourantFriedrichsLewyElement<C, S> for StabilizedElement<N>
where
    C: Solid,
    S: PointStiffness<C>,
{
    type Masses = ElementNodalLumpedMasses<N>;
    type Error = FiniteElementError;
    fn squared_frequency_bound(
        &self,
        constitutive_model: &C,
        stiffness: &S,
        nodal_coordinates: &ElementNodalCoordinates<N>,
        nodal_masses: &ElementNodalLumpedMasses<N>,
    ) -> Result<Quantity<RateSquared>, FiniteElementError> {
        Ok(squared_frequency_bound_from(
            self,
            constitutive_model,
            stiffness,
            nodal_coordinates,
            nodal_masses,
        )? + self
            .hourglass_squared_frequency_bound(constitutive_model.shear_modulus(), nodal_masses))
    }
}
