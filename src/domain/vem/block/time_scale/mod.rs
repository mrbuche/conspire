#[cfg(test)]
mod test;

use crate::{
    constitutive::solid::elastic::Elastic,
    domain::{
        ElementModelError,
        parallel::{Workers, reduce_items},
        solid::time_scale::TimeScaleElements,
    },
    math::Quantity,
    units::Time,
    vem::{
        NodalCoordinates, NodalReferenceCoordinates,
        block::{Block, Densities},
        element::{
            mass::LumpedMassVirtualElement, solid::elastic::ElasticVirtualElement,
            time_scale::fastest_time_scale,
        },
    },
};
use core::any::type_name;

impl<C, F, R> TimeScaleElements<3> for Block<C, F, R>
where
    C: Elastic,
    F: ElasticVirtualElement<C> + LumpedMassVirtualElement,
    R: Densities,
    Self: Sync,
{
    fn fastest_time_scale(
        &self,
        workers: Option<&Workers>,
        reference_coordinates: &NodalReferenceCoordinates,
        nodal_coordinates: &NodalCoordinates,
    ) -> Result<Quantity<Time>, ElementModelError> {
        reduce_items(
            workers,
            self.elements().len(),
            Time::seconds(f64::INFINITY),
            |element_index| {
                let element = &self.elements()[element_index];
                let nodes = &self.elements_nodes()[element_index];
                let stiffnesses = element
                    .nodal_stiffnesses(
                        self.constitutive_model(),
                        &Self::element_coordinates(nodal_coordinates, nodes),
                    )
                    .map_err(|error| ElementModelError::new(type_name::<Self>(), error))?;
                let masses = element.nodal_lumped_masses(
                    self.density().at(element_index),
                    &Self::element_coordinates(reference_coordinates, nodes),
                );
                Ok(fastest_time_scale(&stiffnesses, &masses))
            },
            Quantity::min,
        )
    }
}
