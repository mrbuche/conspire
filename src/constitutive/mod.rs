//! Constitutive models.

#[cfg(test)]
pub mod test;

#[cfg(feature = "autodiff")]
pub mod autodiff;

pub mod canonical;
pub mod chemical;
pub mod cohesive;
pub mod diffusion;
pub mod electrical;
pub mod fluid;
pub mod hybrid;
pub mod multiphysics;
pub mod solid;
pub mod thermal;

use crate::math::{
    Scalar, SquareMatrixError, TensorError, assert::AssertionError, integrate::IntegrationError,
    optimize::OptimizationError,
};
use core::{
    error::Error,
    fmt::{self, Debug, Display, Formatter},
};

/// Required methods for constitutive models.
pub trait Constitutive
where
    Self: Clone + Debug,
{
}

/// Something a constitutive model does not provide.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Unavailable {
    /// The tangent stiffness with respect to the deformation gradient.
    TangentStiffnessWrtDeformationGradient,
    /// The stiffnesses of a surface element, which depend on the velocities.
    StiffnessesAtNonzeroVelocity,
}

/// Possible errors encountered in constitutive models.
pub enum ConstitutiveError {
    /// The Jacobian is not positive.
    InvalidJacobian(Scalar),
    /// The extensibility limit of the model was reached.
    MaximumExtensibility,
    /// Something the model does not provide.
    Unavailable(Unavailable),
    /// A tensor operation failed.
    Tensor(TensorError),
    /// A square-matrix operation failed.
    SquareMatrix(SquareMatrixError),
    /// A solver failed.
    Solver(SolverFailure),
    /// An integrator failed.
    Integrator(IntegratorFailure),
    /// The coupled return map did not converge in this many steps.
    ReturnMapDidNotConverge(usize),
    /// The times of a load must be at least two and increasing.
    InvalidTimes,
    /// The adaptive step fell below the floor.
    StepBelowFloor,
}

/// How a solver failed, without the text its error carried.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SolverFailure {
    /// The maximum number of steps was reached.
    MaximumStepsReached(usize),
    /// The solution is not a minimum.
    NotMinimum,
    /// A matrix was singular.
    SingularMatrix,
    /// A matrix was not symmetric.
    UnsymmetricMatrix,
    /// Another failure, such as one of a nested routine.
    Other,
}

/// How an integrator failed, without the text its error carried.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum IntegratorFailure {
    InconsistentInitialConditions,
    InitialTimeNotLessThanFinalTime,
    InvalidNewmarkParameters(Scalar, Scalar),
    InvalidSafetyFactor(Scalar),
    LengthTimeLessThanTwo,
    MinimumStepSizeReached(Scalar),
    TimeStepNotSet(Scalar, Scalar),
    UnstableTimeStep(Scalar, Scalar),
    /// Another failure, such as one of a nested routine.
    Other,
}

impl ConstitutiveError {
    /// A compact code for the kind of error, for devices that record only that.
    pub const fn code(&self) -> u8 {
        match self {
            Self::InvalidJacobian(_) => 1,
            Self::MaximumExtensibility => 2,
            Self::Unavailable(_) => 3,
            Self::Tensor(_) => 4,
            Self::SquareMatrix(_) => 5,
            Self::Solver(_) => 6,
            Self::Integrator(_) => 7,
            Self::ReturnMapDidNotConverge(_) => 8,
            Self::InvalidTimes => 9,
            Self::StepBelowFloor => 10,
        }
    }
}

impl PartialEq for ConstitutiveError {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::InvalidJacobian(a), Self::InvalidJacobian(b)) => a == b,
            (Self::MaximumExtensibility, Self::MaximumExtensibility) => true,
            (Self::Unavailable(a), Self::Unavailable(b)) => a == b,
            (Self::Tensor(a), Self::Tensor(b)) => a == b,
            (Self::SquareMatrix(a), Self::SquareMatrix(b)) => a == b,
            (Self::Solver(a), Self::Solver(b)) => a == b,
            (Self::Integrator(a), Self::Integrator(b)) => a == b,
            (Self::ReturnMapDidNotConverge(a), Self::ReturnMapDidNotConverge(b)) => a == b,
            (Self::InvalidTimes, Self::InvalidTimes)
            | (Self::StepBelowFloor, Self::StepBelowFloor) => true,
            _ => false,
        }
    }
}

impl From<TensorError> for ConstitutiveError {
    fn from(error: TensorError) -> Self {
        Self::Tensor(error)
    }
}

impl From<SquareMatrixError> for ConstitutiveError {
    fn from(error: SquareMatrixError) -> Self {
        Self::SquareMatrix(error)
    }
}

impl From<ConstitutiveError> for OptimizationError {
    fn from(error: ConstitutiveError) -> Self {
        Self::Intermediate(error.to_string())
    }
}

impl From<IntegrationError> for ConstitutiveError {
    fn from(error: IntegrationError) -> Self {
        use IntegratorFailure as F;
        Self::Integrator(match error {
            IntegrationError::InconsistentInitialConditions => F::InconsistentInitialConditions,
            IntegrationError::InitialTimeNotLessThanFinalTime => F::InitialTimeNotLessThanFinalTime,
            IntegrationError::InvalidNewmarkParameters(beta, gamma) => {
                F::InvalidNewmarkParameters(beta, gamma)
            }
            IntegrationError::InvalidSafetyFactor(safety) => F::InvalidSafetyFactor(safety),
            IntegrationError::LengthTimeLessThanTwo => F::LengthTimeLessThanTwo,
            IntegrationError::MinimumStepSizeReached(dt, _)
            | IntegrationError::MinimumStepSizeUpstream(dt, ..) => F::MinimumStepSizeReached(dt),
            IntegrationError::TimeStepNotSet(t0, tf, _) => F::TimeStepNotSet(t0, tf),
            IntegrationError::UnstableTimeStep(a, b, _) => F::UnstableTimeStep(a, b),
            IntegrationError::Intermediate(_) | IntegrationError::Upstream(..) => F::Other,
        })
    }
}

impl From<OptimizationError> for ConstitutiveError {
    fn from(error: OptimizationError) -> Self {
        Self::Solver(match error {
            OptimizationError::MaximumStepsReached(steps, _) => {
                SolverFailure::MaximumStepsReached(steps)
            }
            OptimizationError::NotMinimum(..) => SolverFailure::NotMinimum,
            OptimizationError::SingularMatrix => SolverFailure::SingularMatrix,
            OptimizationError::UnsymmetricMatrix => SolverFailure::UnsymmetricMatrix,
            OptimizationError::Intermediate(_) | OptimizationError::Upstream(..) => {
                SolverFailure::Other
            }
        })
    }
}

impl Display for ConstitutiveError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidJacobian(jacobian) => write!(f, "Invalid Jacobian: {jacobian:.6e}."),
            Self::MaximumExtensibility => write!(f, "Maximum extensibility reached."),
            Self::Unavailable(Unavailable::TangentStiffnessWrtDeformationGradient) => write!(
                f,
                "The tangent stiffness with respect to the deformation gradient is not available."
            ),
            Self::Unavailable(Unavailable::StiffnessesAtNonzeroVelocity) => write!(
                f,
                "The stiffnesses of a surface element are only available at zero velocities, \
                as its deformation gradient rate depends on the coordinates."
            ),
            Self::Tensor(error) => write!(f, "{error}"),
            Self::SquareMatrix(error) => write!(f, "{error}"),
            Self::Solver(SolverFailure::MaximumStepsReached(steps)) => {
                write!(f, "Maximum number of steps ({steps}) reached.")
            }
            Self::Solver(SolverFailure::NotMinimum) => {
                write!(f, "The obtained solution is not a minimum.")
            }
            Self::Solver(SolverFailure::SingularMatrix) => write!(f, "Matrix is singular."),
            Self::Solver(SolverFailure::UnsymmetricMatrix) => write!(f, "Matrix is not symmetric."),
            Self::Solver(SolverFailure::Other) => write!(f, "The solver failed."),
            Self::Integrator(failure) => write!(f, "The integrator failed: {failure:?}."),
            Self::ReturnMapDidNotConverge(steps) => write!(
                f,
                "The coupled return mapping did not converge in {steps} steps."
            ),
            Self::InvalidTimes => write!(f, "The times must be at least two and increasing."),
            Self::StepBelowFloor => write!(f, "The adaptive step fell below the floor."),
        }
    }
}

impl Debug for ConstitutiveError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "\n{self}\n")
    }
}

impl Error for ConstitutiveError {}

impl From<ConstitutiveError> for AssertionError {
    fn from(error: ConstitutiveError) -> Self {
        Self {
            message: error.to_string(),
        }
    }
}

impl From<ConstitutiveError> for String {
    fn from(error: ConstitutiveError) -> Self {
        error.to_string()
    }
}
