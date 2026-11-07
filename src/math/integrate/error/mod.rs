#[cfg(test)]
mod test;

use crate::math::{Scalar, assert::AssertionError};
use core::fmt;
use core::fmt::{Debug, Display};

/// Possible errors encountered when integrating.
pub enum IntegrationError {
    InconsistentInitialConditions,
    InitialTimeNotLessThanFinalTime,
    InvalidNewmarkParameters(Scalar, Scalar),
    InvalidSafetyFactor(Scalar),
    Intermediate(String),
    LengthTimeLessThanTwo,
    MinimumStepSizeReached(Scalar, String),
    MinimumStepSizeUpstream(Scalar, String, String),
    TimeStepNotSet(Scalar, Scalar, String),
    UnstableTimeStep(Scalar, Scalar, String),
    Upstream(String, String),
}

impl IntegrationError {
    pub fn upstream(error: impl Display, context: &(impl Debug + ?Sized)) -> Self {
        Self::Upstream(format!("{error}"), format!("{context:?}"))
    }
}

impl From<String> for IntegrationError {
    fn from(error: String) -> Self {
        Self::Intermediate(error)
    }
}

impl fmt::Display for IntegrationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InconsistentInitialConditions => {
                write!(
                    f,
                    "The initial condition z_0 is not consistent with g(t_0, y_0)."
                )
            }
            Self::InitialTimeNotLessThanFinalTime => {
                write!(f, "The initial time must precede the final time.")
            }
            Self::InvalidNewmarkParameters(beta, gamma) => {
                write!(
                    f,
                    "The Newmark parameters (beta = {beta:?}, gamma = {gamma:?}) must satisfy 2 beta >= gamma >= 1/2 to be unconditionally stable."
                )
            }
            Self::InvalidSafetyFactor(safety) => {
                write!(f, "The safety factor ({safety:?}) must be in (0, 1].")
            }
            Self::Intermediate(message) => write!(f, "{}", message),
            Self::LengthTimeLessThanTwo => {
                write!(f, "The time must contain at least two entries.")
            }
            Self::MinimumStepSizeReached(dt_min, integrator) => {
                write!(
                    f,
                    "Minimum time step ({dt_min:?}) reached.\n\
                    In integrator: {integrator}."
                )
            }
            Self::MinimumStepSizeUpstream(dt_min, error, integrator) => {
                write!(
                    f,
                    "{error}\n\
                    Causing error: Minimum time step ({dt_min:?}) reached.\n\
                    In integrator: {integrator}."
                )
            }
            Self::TimeStepNotSet(t0, tf, integrator) => {
                write!(
                    f,
                    "A positive time step must be set within [{t0:?}, {tf:?}].\n\
                    In integrator: {integrator}."
                )
            }
            Self::UnstableTimeStep(dt, limit, integrator) => {
                write!(
                    f,
                    "The time step ({dt:?}) exceeds the stability limit ({limit:?}).\n\
                    In integrator: {integrator}."
                )
            }
            Self::Upstream(error, integrator) => {
                write!(
                    f,
                    "{error}\n\
                    In integrator: {integrator}."
                )
            }
        }
    }
}

impl fmt::Debug for IntegrationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "\n{self}\n")
    }
}

impl core::error::Error for IntegrationError {}

impl From<IntegrationError> for String {
    fn from(error: IntegrationError) -> Self {
        format!("{}", error)
    }
}

impl From<IntegrationError> for AssertionError {
    fn from(error: IntegrationError) -> Self {
        AssertionError {
            message: error.to_string(),
        }
    }
}
