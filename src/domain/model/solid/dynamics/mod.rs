#[cfg(test)]
pub(crate) mod test;

mod explicit;
mod implicit;
pub(crate) mod motion;

pub use explicit::{
    Checked, Consistent, CourantFriedrichsLewy, ElasticExplicitDynamics, ElementPowerIteration,
    Execution, Lumped, MassChoice, ModelBound, Stability, TimeScaleEstimate, Unchecked,
};
pub use implicit::ElasticImplicitDynamics;

use crate::{
    domain::{NodalAccelerationsHistory, NodalCoordinatesHistory, NodalVelocitiesHistory},
    math::integrate::Times,
};

type Solution = (
    Times,
    NodalCoordinatesHistory<3>,
    NodalVelocitiesHistory<3>,
    NodalAccelerationsHistory<3>,
);
