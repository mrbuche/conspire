#[cfg(test)]
mod test;

use crate::{
    constitutive::{ConstitutiveError, diffusion::Diffusion, thermal::Thermal},
    math::{IDENTITY_00, Quantity, Tensor},
    mechanics::{HeatFlux, HeatFluxTangent, TemperatureGradient},
    units::{Power, PowerPerLengthTemperature, PowerTemperatureDensity},
};

/// The Fourier thermal conduction constitutive model.
///
/// **Parameters**
/// - The thermal conductivity $`k`$.
///
/// **External variables**
/// - The temperature gradient $`\nabla T`$.
///
/// **Internal variables**
/// - None.
#[derive(Clone, Debug)]
pub struct Fourier {
    /// The thermal conductivity $`k`$.
    pub thermal_conductivity: Quantity<PowerPerLengthTemperature>,
}

impl Fourier {
    fn thermal_conductivity(&self) -> Quantity<PowerPerLengthTemperature> {
        self.thermal_conductivity
    }
}

impl Thermal for Fourier {}

impl Diffusion for Fourier {
    type Balance = Power;
    /// ```math
    /// u(\nabla T) = \frac{1}{2}k\nabla T\cdot\nabla T
    /// ```
    fn potential(
        &self,
        temperature_gradient: &TemperatureGradient,
    ) -> Result<Quantity<PowerTemperatureDensity>, ConstitutiveError> {
        Ok(temperature_gradient.norm_squared() * self.thermal_conductivity() * 0.5)
    }
    /// ```math
    /// \mathbf{q}(\nabla T) = -k\nabla T
    /// ```
    fn flux(
        &self,
        temperature_gradient: &TemperatureGradient,
    ) -> Result<HeatFlux, ConstitutiveError> {
        Ok(temperature_gradient * -self.thermal_conductivity())
    }
    /// ```math
    /// \frac{\partial\mathbf{q}}{\partial\nabla T} = -k\mathbf{I}
    /// ```
    fn tangent(
        &self,
        _temperature_gradient: &TemperatureGradient,
    ) -> Result<HeatFluxTangent, ConstitutiveError> {
        Ok(IDENTITY_00 * -self.thermal_conductivity())
    }
}
