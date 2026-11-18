//! Electrical conduction constitutive models.
//!
//! ---
//!
//! Electrical conduction constitutive models are defined by a relation for the current density as a function of the potential gradient, and by the potential from which that current density derives.

#[cfg(test)]
pub mod test;

mod ohmic;

use crate::{
    constitutive::{ConstitutiveError, diffusion::Diffusion, electrical::Electrical},
    mechanics::{CurrentDensity, CurrentDensityTangent, ElectricField},
    units::Current,
};

pub use ohmic::Ohmic;

/// Required methods for electrical conduction constitutive models.
pub trait ElectricalConduction
where
    Self: Electrical + Diffusion<Balance = Current>,
{
    /// Calculates and returns the current density.
    fn current_density(
        &self,
        potential_gradient: &ElectricField,
    ) -> Result<CurrentDensity, ConstitutiveError> {
        self.flux(potential_gradient)
    }
    /// Calculates and returns the tangent to the current density.
    fn current_density_tangent(
        &self,
        potential_gradient: &ElectricField,
    ) -> Result<CurrentDensityTangent, ConstitutiveError> {
        self.tangent(potential_gradient)
    }
}

impl<C> ElectricalConduction for C where C: Electrical + Diffusion<Balance = Current> {}
