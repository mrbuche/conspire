//! Where the nodal forces of a run of explicit dynamics are computed.

#[cfg(all(test, feature = "cbm"))]
mod test;

use super::ElasticExplicitDynamics;
use crate::{
    domain::{
        ElementModelError, Model, NodalAccelerations, NodalCoordinates,
        boundary::LoadAt,
        solid::{
            NodalForcesSolid,
            elastic::{ParallelElasticElements, Workers},
            mass::InverseMass,
        },
    },
    math::{Quantity, Tensor, integrate::IntegrationError},
    units::Time,
};

pub use crate::math::Execution;

/// The threads and storage of one run.
pub(super) enum Run<S> {
    Serial,
    Parallel(Workers, S),
}

impl Execution {
    pub(super) fn start<B>(self, model: &Model<B, 3>) -> Result<Run<B::Scratch>, IntegrationError>
    where
        B: ParallelElasticElements<3>,
    {
        let threads = match self {
            Self::Serial => return Ok(Run::Serial),
            Self::Threads(_) => self.threads(),
        };
        Ok(Run::Parallel(
            Workers::new(threads).map_err(IntegrationError::Intermediate)?,
            model.scratch(),
        ))
    }
}

impl<B> Model<B, 3>
where
    B: ParallelElasticElements<3>,
{
    pub(super) fn nodal_accelerations_in(
        &self,
        run: &mut Run<B::Scratch>,
        nodal_coordinates: &NodalCoordinates<3>,
        time: Quantity<Time>,
        applied_loads: &impl LoadAt<NodalCoordinates<3>, NodalForcesSolid<3>>,
        masses: &impl InverseMass<3>,
    ) -> Result<NodalAccelerations<3>, ElementModelError> {
        match run {
            Run::Serial => self.nodal_accelerations(nodal_coordinates, time, applied_loads, masses),
            Run::Parallel(workers, scratch) => {
                let mut unbalanced_forces = NodalForcesSolid::zero(nodal_coordinates.len());
                self.nodal_forces_parallel_into(
                    scratch,
                    workers,
                    nodal_coordinates,
                    &mut unbalanced_forces,
                )?;
                applied_loads
                    .subtract_at(time, nodal_coordinates, &mut unbalanced_forces)
                    .map_err(|error| ElementModelError::new("the applied loads", error))?;
                Ok(masses.nodal_accelerations(&unbalanced_forces))
            }
        }
    }
}
