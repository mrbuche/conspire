use crate::{
    constitutive::solid::elastic::Elastic,
    domain::{ItemError, Source},
    fem::{
        ElementModelError, NodalCoordinates, NodalReferenceCoordinates,
        block::{Block, Densities},
        element::{
            ElementNodalCoordinates, FiniteElementError,
            mass::{ElementNodalLumpedMasses, LumpedMassFiniteElement},
            solid::elastic::ElasticFiniteElement,
            time_scale::{CourantFriedrichsLewyElement, fastest_time_scale},
        },
        solid::time_scale::{CourantFriedrichsLewyElements, TimeScaleElements},
    },
    math::Quantity,
    units::{RateSquared, Time},
};
use core::any::type_name;

impl<C, F, R, H, const G: usize, const M: usize, const N: usize, const P: usize>
    TimeScaleElements<3> for Block<C, F, G, M, N, P, R, H>
where
    C: Elastic,
    F: ElasticFiniteElement<C, G, M, N, P> + LumpedMassFiniteElement<G, M, N, P>,
    R: Densities<G>,
{
    fn fastest_time_scale(
        &self,
        _reference_coordinates: &NodalReferenceCoordinates<3>,
        nodal_coordinates: &NodalCoordinates<3>,
    ) -> Result<Quantity<Time>, ElementModelError> {
        self.elements()
            .iter()
            .zip(self.connectivity())
            .enumerate()
            .try_fold(
                Time::seconds(f64::INFINITY),
                |fastest, (element_index, (element, nodes))| {
                    let stiffnesses = element.nodal_stiffnesses(
                        self.constitutive_model(),
                        &Self::element_coordinates(nodal_coordinates, nodes),
                    )?;
                    let masses = element.nodal_lumped_masses(&self.density().at(element_index));
                    Ok::<_, FiniteElementError>(
                        fastest.min(fastest_time_scale(&stiffnesses, &masses)),
                    )
                },
            )
            .map_err(|error| ElementModelError::new(type_name::<Self>(), error))
    }
}

impl<C, F, R, H, S, const G: usize, const N: usize> CourantFriedrichsLewyElements<S>
    for Block<C, F, G, 3, N, N, R, H>
where
    C: Elastic,
    F: LumpedMassFiniteElement<G, 3, N, N>
        + CourantFriedrichsLewyElement<
            C,
            S,
            Coordinates = ElementNodalCoordinates<N>,
            Masses = ElementNodalLumpedMasses<N>,
            Error = FiniteElementError,
        >,
    R: Densities<G>,
{
    fn courant_friedrichs_lewy_time_scale(
        &self,
        stiffness: &S,
        nodal_coordinates: &NodalCoordinates<3>,
    ) -> Result<Quantity<Time>, ElementModelError> {
        let mut largest = Quantity::<RateSquared>::default();
        for (element_index, (element, nodes)) in
            self.elements().iter().zip(self.connectivity()).enumerate()
        {
            let squared_frequency = element
                .squared_frequency_bound(
                    self.constitutive_model(),
                    stiffness,
                    &Self::element_coordinates(nodal_coordinates, nodes),
                    &element.nodal_lumped_masses(&self.density().at(element_index)),
                )
                .map_err(|source| {
                    ElementModelError::new(
                        type_name::<Self>(),
                        ItemError::new(element_index, source),
                    )
                })?;
            if !squared_frequency.value().is_finite() {
                return Err(ElementModelError::new(
                    type_name::<Self>(),
                    ItemError::new(
                        element_index,
                        Source::Text(
                            "The bound on the frequency of the element is not finite.".to_string(),
                        ),
                    ),
                ));
            }
            largest = largest.max(squared_frequency);
        }
        Ok(1.0 / largest.sqrt())
    }
}
