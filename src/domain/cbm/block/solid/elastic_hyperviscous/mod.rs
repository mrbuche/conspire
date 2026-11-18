use super::super::Block;
use crate::cbm::node::solid::elastic_hyperviscous::ElasticHyperviscousElement;
use crate::domain::AtItems;
use crate::{
    constitutive::solid::elastic_hyperviscous::ElasticHyperviscous,
    domain::{ElementModelError, NodalCoordinates, NodalVelocities},
    math::Quantity,
    units::Power,
};
use core::any::type_name;

pub use crate::domain::solid::elastic_hyperviscous::ElasticHyperviscousElements;

impl<C, R> ElasticHyperviscousElements<3> for Block<C, R>
where
    C: ElasticHyperviscous,
{
    fn viscous_dissipation(
        &self,
        nodal_coordinates: &NodalCoordinates<3>,
        nodal_velocities: &NodalVelocities<3>,
    ) -> Result<Quantity<Power>, ElementModelError> {
        self.nodes
            .iter()
            .map(|node| {
                node.viscous_dissipation(
                    &self.constitutive_model,
                    nodal_coordinates,
                    nodal_velocities,
                )
            })
            .sum_at_items()
            .map_err(|error| ElementModelError::new(type_name::<Self>(), error))
    }
    fn dissipation_potential(
        &self,
        nodal_coordinates: &NodalCoordinates<3>,
        nodal_velocities: &NodalVelocities<3>,
    ) -> Result<Quantity<Power>, ElementModelError> {
        self.nodes
            .iter()
            .map(|node| {
                node.dissipation_potential(
                    &self.constitutive_model,
                    nodal_coordinates,
                    nodal_velocities,
                )
            })
            .sum_at_items()
            .map_err(|error| ElementModelError::new(type_name::<Self>(), error))
    }
}
