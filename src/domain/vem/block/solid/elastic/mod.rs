use crate::domain::ItemError;
use crate::{
    constitutive::solid::elastic::Elastic,
    domain::{ElementModelError, solid::elastic::ElasticItems},
    vem::{
        NodalCoordinates,
        block::{
            Block,
            element::solid::{
                ElementNodalForcesSolid, ElementNodalStiffnessesSolid,
                elastic::ElasticVirtualElement,
            },
        },
    },
};
use core::any::type_name;

impl<C, F, R> ElasticItems<3> for Block<C, F, R>
where
    C: Elastic,
    F: ElasticVirtualElement<C>,
{
    type Forces = ElementNodalForcesSolid;
    type Stiffnesses = ElementNodalStiffnessesSolid;
    fn forces_of(
        &self,
        item: usize,
        nodal_coordinates: &NodalCoordinates,
    ) -> Result<ElementNodalForcesSolid, ElementModelError> {
        self.elements()[item]
            .nodal_forces(
                self.constitutive_model(),
                &Self::element_coordinates(nodal_coordinates, &self.elements_nodes()[item]),
            )
            .map_err(|error| {
                ElementModelError::new(type_name::<Self>(), ItemError::new(item, error))
            })
    }
    fn stiffnesses_of(
        &self,
        item: usize,
        nodal_coordinates: &NodalCoordinates,
    ) -> Result<ElementNodalStiffnessesSolid, ElementModelError> {
        self.elements()[item]
            .nodal_stiffnesses(
                self.constitutive_model(),
                &Self::element_coordinates(nodal_coordinates, &self.elements_nodes()[item]),
            )
            .map_err(|error| {
                ElementModelError::new(type_name::<Self>(), ItemError::new(item, error))
            })
    }
}
