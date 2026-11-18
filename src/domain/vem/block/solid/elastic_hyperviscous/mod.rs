use crate::domain::AtItems;
use crate::{
    constitutive::solid::elastic_hyperviscous::ElasticHyperviscous,
    domain::{
        ElementModelError,
        solid::{
            elastic_hyperviscous::ElasticHyperviscousElements, viscoelastic::ViscoelasticElements,
        },
    },
    math::Quantity,
    units::Power,
    vem::{
        NodalCoordinates, NodalVelocities,
        block::{Block, element::solid::elastic_hyperviscous::ElasticHyperviscousVirtualElement},
    },
};
use core::any::type_name;

impl<C, F, R> ElasticHyperviscousElements<3> for Block<C, F, R>
where
    C: ElasticHyperviscous,
    F: ElasticHyperviscousVirtualElement<C>,
    Self: ViscoelasticElements<3>,
{
    fn viscous_dissipation(
        &self,
        nodal_coordinates: &NodalCoordinates,
        nodal_velocities: &NodalVelocities,
    ) -> Result<Quantity<Power>, ElementModelError> {
        self.elements()
            .iter()
            .zip(self.elements_nodes())
            .map(|(element, nodes)| {
                element.viscous_dissipation(
                    self.constitutive_model(),
                    &Self::element_coordinates(nodal_coordinates, nodes),
                    &Self::element_coordinates(nodal_velocities, nodes),
                )
            })
            .sum_at_items()
            .map_err(|error| ElementModelError::new(type_name::<Self>(), error))
    }
    fn dissipation_potential(
        &self,
        nodal_coordinates: &NodalCoordinates,
        nodal_velocities: &NodalVelocities,
    ) -> Result<Quantity<Power>, ElementModelError> {
        self.elements()
            .iter()
            .zip(self.elements_nodes())
            .map(|(element, nodes)| {
                element.dissipation_potential(
                    self.constitutive_model(),
                    &Self::element_coordinates(nodal_coordinates, nodes),
                    &Self::element_coordinates(nodal_velocities, nodes),
                )
            })
            .sum_at_items()
            .map_err(|error| ElementModelError::new(type_name::<Self>(), error))
    }
}
