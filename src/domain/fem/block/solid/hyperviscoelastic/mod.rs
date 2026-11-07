use crate::domain::AtItems;
use crate::{
    constitutive::solid::hyperviscoelastic::Hyperviscoelastic,
    fem::{
        ElementModelError, NodalCoordinates,
        block::{Block, element::solid::hyperviscoelastic::HyperviscoelasticFiniteElement},
        solid::{
            elastic_hyperviscous::ElasticHyperviscousElements,
            hyperviscoelastic::HyperviscoelasticElements,
        },
    },
    math::Quantity,
    units::Energy,
};
use core::any::type_name;

impl<C, F, const G: usize, const M: usize, const N: usize, const P: usize, R, H>
    HyperviscoelasticElements<3> for Block<C, F, G, M, N, P, R, H>
where
    C: Hyperviscoelastic,
    F: HyperviscoelasticFiniteElement<C, G, M, N, P>,
    Self: ElasticHyperviscousElements<3>,
{
    fn helmholtz_free_energy(
        &self,
        nodal_coordinates: &NodalCoordinates<3>,
    ) -> Result<Quantity<Energy>, ElementModelError> {
        self.elements()
            .iter()
            .zip(self.connectivity())
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
