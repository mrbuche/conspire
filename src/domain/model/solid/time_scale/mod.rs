use crate::{
    domain::{
        Blocks, ElementModelError, Model, NodalCoordinates, NodalReferenceCoordinates,
        block::element::Elements,
        parallel::Workers,
        stability::{CourantFriedrichsLewy, ElementPowerIteration},
    },
    math::Quantity,
    units::Time,
};

/// Elements that report how fast their motion can be.
#[cfg_attr(not(any(feature = "fem", feature = "vem")), allow(dead_code))]
pub trait TimeScaleElements<const D: usize>
where
    Self: Elements,
{
    /// The shortest time scale among the elements at the given coordinates, the
    /// reciprocal of the highest angular frequency, which bounds a stable explicit time step.
    fn fastest_time_scale(
        &self,
        workers: Option<&Workers>,
        reference_coordinates: &NodalReferenceCoordinates<D>,
        nodal_coordinates: &NodalCoordinates<D>,
    ) -> Result<Quantity<Time>, ElementModelError>;
}

impl<B, const D: usize> TimeScaleElements<D> for Model<B, D>
where
    B: TimeScaleElements<D>,
{
    fn fastest_time_scale(
        &self,
        workers: Option<&Workers>,
        reference_coordinates: &NodalReferenceCoordinates<D>,
        nodal_coordinates: &NodalCoordinates<D>,
    ) -> Result<Quantity<Time>, ElementModelError> {
        self.blocks
            .fastest_time_scale(workers, reference_coordinates, nodal_coordinates)
    }
}

impl<B1, B2, const D: usize> TimeScaleElements<D> for Blocks<B1, B2>
where
    B1: TimeScaleElements<D>,
    B2: TimeScaleElements<D>,
{
    fn fastest_time_scale(
        &self,
        workers: Option<&Workers>,
        reference_coordinates: &NodalReferenceCoordinates<D>,
        nodal_coordinates: &NodalCoordinates<D>,
    ) -> Result<Quantity<Time>, ElementModelError> {
        Ok(self
            .0
            .fastest_time_scale(workers, reference_coordinates, nodal_coordinates)?
            .min(
                self.1
                    .fastest_time_scale(workers, reference_coordinates, nodal_coordinates)
                    .map_err(|error| error.in_block(self.0.number_of_blocks()))?,
            ))
    }
}

/// A way to estimate the fastest time scale of the motion of a model.
pub trait TimeScaleEstimate<M> {
    /// The fastest time scale of the model, which bounds a stable explicit time step.
    fn time_scale(
        &self,
        model: &M,
        workers: Option<&Workers>,
        nodal_coordinates: &NodalCoordinates<3>,
    ) -> Result<Quantity<Time>, ElementModelError>;
}

impl<B> TimeScaleEstimate<Model<B, 3>> for ElementPowerIteration
where
    B: TimeScaleElements<3>,
{
    fn time_scale(
        &self,
        model: &Model<B, 3>,
        workers: Option<&Workers>,
        nodal_coordinates: &NodalCoordinates<3>,
    ) -> Result<Quantity<Time>, ElementModelError> {
        model.fastest_time_scale(workers, &model.coordinates, nodal_coordinates)
    }
}

/// Elements that bound how fast their motion can be from their integration points, with the
/// stiffness at each as `S` has it.
#[cfg_attr(not(feature = "fem"), allow(dead_code))]
pub trait CourantFriedrichsLewyElements<S>
where
    Self: Elements,
{
    /// A time scale that is no longer than that of any element at the given coordinates, which
    /// bounds a stable explicit time step.
    fn courant_friedrichs_lewy_time_scale(
        &self,
        stiffness: &S,
        workers: Option<&Workers>,
        nodal_coordinates: &NodalCoordinates<3>,
    ) -> Result<Quantity<Time>, ElementModelError>;
}

impl<B1, B2, S> CourantFriedrichsLewyElements<S> for Blocks<B1, B2>
where
    B1: CourantFriedrichsLewyElements<S>,
    B2: CourantFriedrichsLewyElements<S>,
{
    fn courant_friedrichs_lewy_time_scale(
        &self,
        stiffness: &S,
        workers: Option<&Workers>,
        nodal_coordinates: &NodalCoordinates<3>,
    ) -> Result<Quantity<Time>, ElementModelError> {
        Ok(self
            .0
            .courant_friedrichs_lewy_time_scale(stiffness, workers, nodal_coordinates)?
            .min(
                self.1
                    .courant_friedrichs_lewy_time_scale(stiffness, workers, nodal_coordinates)
                    .map_err(|error| error.in_block(self.0.number_of_blocks()))?,
            ))
    }
}

impl<B, S> TimeScaleEstimate<Model<B, 3>> for CourantFriedrichsLewy<S>
where
    B: CourantFriedrichsLewyElements<S>,
{
    fn time_scale(
        &self,
        model: &Model<B, 3>,
        workers: Option<&Workers>,
        nodal_coordinates: &NodalCoordinates<3>,
    ) -> Result<Quantity<Time>, ElementModelError> {
        model
            .blocks
            .courant_friedrichs_lewy_time_scale(&self.stiffness, workers, nodal_coordinates)
    }
}
