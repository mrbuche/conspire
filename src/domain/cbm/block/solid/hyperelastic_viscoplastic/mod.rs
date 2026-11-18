use super::super::Block;
use crate::cbm::node::solid::hyperelastic_viscoplastic::HyperelasticViscoplasticElement;
use crate::domain::AtItems;
use crate::{
    constitutive::solid::hyperelastic_viscoplastic::HyperelasticViscoplastic,
    domain::{
        ElementModelError, NodalCoordinates, block::solid::viscoplastic::ViscoplasticStateVariables,
    },
    math::{Differentiable, Quantity, Tensor},
    units::Energy,
};
use core::any::type_name;

pub use crate::domain::solid::hyperelastic_viscoplastic::HyperelasticViscoplasticElements;

impl<C, R, Y> HyperelasticViscoplasticElements<ViscoplasticStateVariables<1, Y>, 3> for Block<C, R>
where
    C: HyperelasticViscoplastic<Y>,
    Y: Differentiable + Tensor,
{
    fn helmholtz_free_energy(
        &self,
        nodal_coordinates: &NodalCoordinates<3>,
        state_variables: &ViscoplasticStateVariables<1, Y>,
    ) -> Result<Quantity<Energy>, ElementModelError> {
        self.nodes
            .iter()
            .zip(state_variables)
            .map(|(node, state_variables_node)| {
                node.helmholtz_free_energy(
                    &self.constitutive_model,
                    nodal_coordinates,
                    state_variables_node,
                )
            })
            .sum_at_items()
            .map_err(|error| ElementModelError::new(type_name::<Self>(), error))
    }
}
