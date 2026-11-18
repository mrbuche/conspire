use crate::{
    domain::{
        Blocks, ElementModelError, Model, NodalCoordinates,
        solid::elastic_hyperviscous::ElasticHyperviscousElements,
    },
    math::Quantity,
    units::Energy,
};

pub trait HyperviscoelasticElements<const D: usize>
where
    Self: ElasticHyperviscousElements<D>,
{
    fn helmholtz_free_energy(
        &self,
        nodal_coordinates: &NodalCoordinates<D>,
    ) -> Result<Quantity<Energy>, ElementModelError>;
}

impl<B, const D: usize> HyperviscoelasticElements<D> for Model<B, D>
where
    B: HyperviscoelasticElements<D>,
{
    fn helmholtz_free_energy(
        &self,
        nodal_coordinates: &NodalCoordinates<D>,
    ) -> Result<Quantity<Energy>, ElementModelError> {
        self.blocks.helmholtz_free_energy(nodal_coordinates)
    }
}

impl<B1, B2, const D: usize> HyperviscoelasticElements<D> for Blocks<B1, B2>
where
    B1: HyperviscoelasticElements<D>,
    B2: HyperviscoelasticElements<D>,
{
    fn helmholtz_free_energy(
        &self,
        nodal_coordinates: &NodalCoordinates<D>,
    ) -> Result<Quantity<Energy>, ElementModelError> {
        Ok(self
            .0
            .helmholtz_free_energy(nodal_coordinates)
            .map_err(|error| error.in_block(0))?
            + self
                .1
                .helmholtz_free_energy(nodal_coordinates)
                .map_err(|error| error.in_block(self.0.number_of_blocks()))?)
    }
}
