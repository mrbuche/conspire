#[cfg(test)]
mod test;

use super::{Block, density::Densities};
use crate::cbm::node::mass::NodalMass;
use crate::{
    constitutive::solid::elastic::Elastic,
    domain::{
        ElementModelError, NodalCoordinates, NodalReferenceCoordinates,
        block::element::solid::elastic::ElasticElement,
        parallel::{Workers, reduce_items},
        solid::time_scale::TimeScaleElements,
        time_scale::{largest_eigenvalue, time_scale_from_eigenvalue},
    },
    math::{Quantity, Scalar},
    units::Time,
};
use core::any::type_name;

impl<C, R> TimeScaleElements<3> for Block<C, R>
where
    C: Elastic,
    R: Densities,
    Self: Sync,
{
    fn fastest_time_scale(
        &self,
        workers: Option<&Workers>,
        _reference_coordinates: &NodalReferenceCoordinates<3>,
        nodal_coordinates: &NodalCoordinates<3>,
    ) -> Result<Quantity<Time>, ElementModelError> {
        reduce_items(
            workers,
            self.nodes.len(),
            0.0,
            |index| {
                let node = &self.nodes[index];
                let stiffnesses = node
                    .nodal_stiffnesses(&self.constitutive_model, nodal_coordinates)
                    .map_err(|error| ElementModelError::new(type_name::<Self>(), error))?;
                let masses: Vec<Scalar> = node
                    .neighbors()
                    .iter()
                    .flat_map(|&neighbor| {
                        let share = self.nodes[neighbor]
                            .nodal_mass(self.density.at(neighbor))
                            .value()
                            / self.nodes[neighbor].neighbors().len() as Scalar;
                        [share; 3]
                    })
                    .collect();
                Ok(largest_eigenvalue(
                    masses.len(),
                    |row, column| stiffnesses[row / 3][column / 3][row % 3][column % 3].value(),
                    &masses,
                ))
            },
            Scalar::max,
        )
        .map(time_scale_from_eigenvalue)
    }
}
