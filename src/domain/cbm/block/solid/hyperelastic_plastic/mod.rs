use super::super::Block;
use crate::cbm::node::solid::hyperelastic_plastic::HyperelasticPlasticElement;
use crate::domain::AtItems;
use crate::{
    constitutive::solid::hyperelastic_plastic::HyperelasticPlastic,
    domain::{
        ElementModelError, NodalCoordinates, block::solid::plastic::PlasticStateVariablesField,
    },
    math::Quantity,
    units::Energy,
};
use core::any::type_name;

pub use crate::domain::solid::hyperelastic_plastic::HyperelasticPlasticElements;

impl<C, R> HyperelasticPlasticElements<PlasticStateVariablesField<1>, 3> for Block<C, R>
where
    C: HyperelasticPlastic,
{
    fn helmholtz_free_energy(
        &self,
        nodal_coordinates: &NodalCoordinates<3>,
        state_variables: &PlasticStateVariablesField<1>,
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
