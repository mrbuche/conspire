#[cfg(test)]
mod test;

use crate::{
    constitutive::{ConstitutiveError, chemical::Chemical, diffusion::Diffusion},
    math::{IDENTITY_00, Quantity, Tensor},
    mechanics::{ConcentrationGradient, MolarFlux, MolarFluxTangent},
    units::{Diffusivity, MolarFlowRate, MolarFlowRateConcentrationDensity},
};

/// The Fick chemical diffusion constitutive model.
///
/// **Parameters**
/// - The diffusivity $`D`$.
///
/// **External variables**
/// - The concentration gradient $`\nabla c`$.
///
/// **Internal variables**
/// - None.
#[derive(Clone, Debug)]
pub struct Fick {
    /// The diffusivity $`D`$.
    pub diffusivity: Quantity<Diffusivity>,
}

impl Chemical for Fick {}

impl Diffusion for Fick {
    type Balance = MolarFlowRate;
    /// ```math
    /// u(\nabla c) = \frac{1}{2}D\nabla c\cdot\nabla c
    /// ```
    fn potential(
        &self,
        concentration_gradient: &ConcentrationGradient,
    ) -> Result<Quantity<MolarFlowRateConcentrationDensity>, ConstitutiveError> {
        Ok(concentration_gradient.norm_squared() * self.diffusivity * 0.5)
    }
    /// ```math
    /// \mathbf{J}(\nabla c) = -D\nabla c
    /// ```
    fn flux(
        &self,
        concentration_gradient: &ConcentrationGradient,
    ) -> Result<MolarFlux, ConstitutiveError> {
        Ok(concentration_gradient * -self.diffusivity)
    }
    /// ```math
    /// \frac{\partial\mathbf{J}}{\partial\nabla c} = -D\mathbf{I}
    /// ```
    fn tangent(
        &self,
        _concentration_gradient: &ConcentrationGradient,
    ) -> Result<MolarFluxTangent, ConstitutiveError> {
        Ok(IDENTITY_00 * -self.diffusivity)
    }
}
