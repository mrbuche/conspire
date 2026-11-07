mod arbitrary;
mod efjc;
mod efrc;
mod fjc;
mod frc;
mod ideal;
mod swfjc;
mod ufjc;

/// Single-chain models of polymer statistical thermodynamics.
mod thermodynamics;

pub use arbitrary::{ArbitraryDiscrete, ArbitraryDiscretePotential};
pub use efjc::ExtensibleFreelyJointedChain;
pub use efrc::ExtensibleFreelyRotatingChain;
pub use fjc::FreelyJointedChain;
pub use frc::FreelyRotatingChain;
pub use ideal::IdealChain;
pub use swfjc::SquareWellFreelyJointedChain;
pub use thermodynamics::{
    Configuration, Ensemble, Isometric, Isotensional, IsotensionalExtensible, Legendre, MonteCarlo,
    MonteCarloExtensible, MonteCarloInextensible, Thermodynamics, ThermodynamicsExtensible,
};
pub use ufjc::ArbitraryPotentialFreelyJointedChain;

use crate::math::{Quantity, Scalar, assert::AssertionError};
use crate::units::Length;
use core::fmt;
use core::fmt::{Debug, Display};

pub trait SingleChain
where
    Self: Clone + Debug,
{
    fn link_length(&self) -> Quantity<Length>;
    fn number_of_links(&self) -> u8;
}

pub trait Inextensible
where
    Self: SingleChain,
{
    fn maximum_nondimensional_extension(&self) -> Scalar;
    fn nondimensional_extension_check(
        &self,
        nondimensional_extension: Scalar,
    ) -> Result<(), SingleChainError> {
        if nondimensional_extension.abs() >= self.maximum_nondimensional_extension() {
            Err(SingleChainError::MaximumExtensibility(
                format!("{:?}", self.maximum_nondimensional_extension()),
                format!("{self:?}"),
            ))
        } else {
            Ok(())
        }
    }
}

pub trait Extensible
where
    Self: SingleChain,
{
}

pub enum SingleChainError {
    MaximumExtensibility(String, String),
    Upstream(String, String),
}

impl SingleChainError {
    pub fn upstream(error: impl Display, context: &(impl Debug + ?Sized)) -> Self {
        Self::Upstream(format!("{error}"), format!("{context:?}"))
    }
}

impl From<SingleChainError> for AssertionError {
    fn from(error: SingleChainError) -> Self {
        Self {
            message: error.to_string(),
        }
    }
}

impl From<SingleChainError> for String {
    fn from(error: SingleChainError) -> Self {
        error.to_string()
    }
}

impl fmt::Display for SingleChainError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MaximumExtensibility(maximum_nondimensional_extension, single_chain_model) => {
                write!(
                    f,
                    "Maximum extensibility ({maximum_nondimensional_extension}) reached.\n\
                    In single-chain model: {single_chain_model}."
                )
            }
            Self::Upstream(error, single_chain_model) => write!(
                f,
                "{error}\n\
                In single-chain model: {single_chain_model}."
            ),
        }
    }
}

impl fmt::Debug for SingleChainError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "\n{self}\n")
    }
}

impl core::error::Error for SingleChainError {}
