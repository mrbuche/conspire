use super::super::{Block, node::solid::hyperviscoelastic::HyperviscoelasticElement};
use crate::domain::AtItems;
use crate::{
    constitutive::solid::hyperviscoelastic::Hyperviscoelastic,
    domain::{ElementModelError, NodalCoordinates},
    math::Quantity,
    units::Energy,
};
use core::any::type_name;

pub use crate::domain::solid::hyperviscoelastic::HyperviscoelasticElements;

impl<C, R> HyperviscoelasticElements<3> for Block<C, R>
where
    C: Hyperviscoelastic,
{
    fn helmholtz_free_energy(
        &self,
        nodal_coordinates: &NodalCoordinates<3>,
    ) -> Result<Quantity<Energy>, ElementModelError> {
        self.nodes
            .iter()
            .map(|node| node.helmholtz_free_energy(&self.constitutive_model, nodal_coordinates))
            .sum_at_items()
            .map_err(|error| ElementModelError::new(type_name::<Self>(), error))
    }
}
