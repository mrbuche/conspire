//! Chemical diffusion constitutive models.
//!
//! ---
//!
//! Chemical diffusion constitutive models are defined by a relation for the molar flux as a function of the concentration gradient, and by the potential from which that molar flux derives.

#[cfg(test)]
pub mod test;

mod fick;

use crate::{
    constitutive::{ConstitutiveError, chemical::Chemical, diffusion::Diffusion},
    mechanics::{ConcentrationGradient, MolarFlux, MolarFluxTangent},
    units::MolarFlowRate,
};

pub use fick::Fick;

/// Required methods for chemical diffusion constitutive models.
pub trait ChemicalDiffusion
where
    Self: Chemical + Diffusion<Balance = MolarFlowRate>,
{
    /// Calculates and returns the molar flux.
    fn molar_flux(
        &self,
        concentration_gradient: &ConcentrationGradient,
    ) -> Result<MolarFlux, ConstitutiveError> {
        self.flux(concentration_gradient)
    }
    /// Calculates and returns the tangent to the molar flux.
    fn molar_flux_tangent(
        &self,
        concentration_gradient: &ConcentrationGradient,
    ) -> Result<MolarFluxTangent, ConstitutiveError> {
        self.tangent(concentration_gradient)
    }
}

impl<C> ChemicalDiffusion for C where C: Chemical + Diffusion<Balance = MolarFlowRate> {}
