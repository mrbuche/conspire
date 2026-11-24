//! Constitutive models.

#[cfg(test)]
pub mod test;

pub mod canonical;
pub mod chemical;
pub mod cohesive;
pub mod diffusion;
pub mod electrical;
mod error;
pub mod fluid;
pub mod hybrid;
pub mod multiphysics;
mod parameters;
pub mod solid;
pub mod thermal;

pub use error::{ConstitutiveError, IntegratorFailure, SolverFailure, Unavailable};
pub use parameters::FromParameters;

use core::fmt::Debug;

/// Required methods for constitutive models.
pub trait Constitutive
where
    Self: Clone + Debug,
{
}
