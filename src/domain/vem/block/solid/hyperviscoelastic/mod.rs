use crate::domain::AtItems;
use crate::{
    constitutive::solid::hyperviscoelastic::Hyperviscoelastic,
    domain::{
        ElementModelError,
        solid::{
            elastic_hyperviscous::ElasticHyperviscousElements,
            hyperviscoelastic::HyperviscoelasticElements,
        },
    },
    math::Quantity,
    units::Energy,
    vem::{
        NodalCoordinates, block::Block,
        element::solid::hyperviscoelastic::HyperviscoelasticVirtualElement,
    },
};
use core::any::type_name;

impl<C, F, R> HyperviscoelasticElements<3> for Block<C, F, R>
where
    C: Hyperviscoelastic,
    F: HyperviscoelasticVirtualElement<C>,
    Self: ElasticHyperviscousElements<3>,
{
    fn helmholtz_free_energy(
        &self,
        nodal_coordinates: &NodalCoordinates,
    ) -> Result<Quantity<Energy>, ElementModelError> {
        self.elements()
            .iter()
            .zip(self.elements_nodes())
            .map(|(element, nodes)| {
                element.helmholtz_free_energy(
                    self.constitutive_model(),
                    &Self::element_coordinates(nodal_coordinates, nodes),
                )
            })
            .sum_at_items()
            .map_err(|error| ElementModelError::new(type_name::<Self>(), error))
    }
}
