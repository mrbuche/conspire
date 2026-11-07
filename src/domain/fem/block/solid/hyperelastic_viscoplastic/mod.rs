use crate::domain::AtItems;
use crate::{
    constitutive::solid::hyperelastic_viscoplastic::HyperelasticViscoplastic,
    fem::{
        ElementModelError, NodalCoordinates,
        block::{
            Block,
            element::solid::hyperelastic_viscoplastic::HyperelasticViscoplasticFiniteElement,
            solid::elastic_viscoplastic::ViscoplasticStateVariables,
        },
        solid::{
            elastic_viscoplastic::ElasticViscoplasticElements,
            hyperelastic_viscoplastic::HyperelasticViscoplasticElements,
        },
    },
    math::{Differentiable, Quantity, Tensor},
    units::Energy,
};
use core::any::type_name;

impl<C, F, const G: usize, const M: usize, const N: usize, const P: usize, Y, R, H>
    HyperelasticViscoplasticElements<ViscoplasticStateVariables<G, Y>, 3>
    for Block<C, F, G, M, N, P, R, H>
where
    C: HyperelasticViscoplastic<Y>,
    F: HyperelasticViscoplasticFiniteElement<C, G, M, N, P, Y>,
    Self: ElasticViscoplasticElements<ViscoplasticStateVariables<G, Y>, 3>,
    Y: Differentiable + Tensor,
{
    fn helmholtz_free_energy(
        &self,
        nodal_coordinates: &NodalCoordinates<3>,
        state_variables: &ViscoplasticStateVariables<G, Y>,
    ) -> Result<Quantity<Energy>, ElementModelError> {
        self.elements()
            .iter()
            .zip(self.connectivity())
            .zip(state_variables)
            .map(|((element, nodes), state_variables_element)| {
                element.helmholtz_free_energy(
                    self.constitutive_model(),
                    &Self::element_coordinates(nodal_coordinates, nodes),
                    state_variables_element,
                )
            })
            .sum_at_items()
            .map_err(|error| ElementModelError::new(type_name::<Self>(), error))
    }
}
