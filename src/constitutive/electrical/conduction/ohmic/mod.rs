#[cfg(test)]
mod test;

use crate::{
    constitutive::{ConstitutiveError, diffusion::Diffusion, electrical::Electrical},
    math::{IDENTITY_00, Quantity, Tensor},
    mechanics::{CurrentDensity, CurrentDensityTangent, ElectricField},
    units::{Current, ElectricalConductivity, PowerDensity},
};

/// The Ohmic electrical conduction constitutive model.
///
/// **Parameters**
/// - The electrical conductivity $`\sigma`$.
///
/// **External variables**
/// - The potential gradient $`\nabla\phi`$.
///
/// **Internal variables**
/// - None.
#[derive(Clone, Debug)]
pub struct Ohmic {
    /// The electrical conductivity $`\sigma`$.
    pub electrical_conductivity: Quantity<ElectricalConductivity>,
}

impl Electrical for Ohmic {}

impl Diffusion for Ohmic {
    type Balance = Current;
    /// ```math
    /// u(\nabla\phi) = \frac{1}{2}\sigma\nabla\phi\cdot\nabla\phi
    /// ```
    fn potential(
        &self,
        potential_gradient: &ElectricField,
    ) -> Result<Quantity<PowerDensity>, ConstitutiveError> {
        Ok(potential_gradient.norm_squared() * self.electrical_conductivity * 0.5)
    }
    /// ```math
    /// \mathbf{J}(\nabla\phi) = -\sigma\nabla\phi
    /// ```
    fn flux(
        &self,
        potential_gradient: &ElectricField,
    ) -> Result<CurrentDensity, ConstitutiveError> {
        Ok(potential_gradient * -self.electrical_conductivity)
    }
    /// ```math
    /// \frac{\partial\mathbf{J}}{\partial\nabla\phi} = -\sigma\mathbf{I}
    /// ```
    fn tangent(
        &self,
        _potential_gradient: &ElectricField,
    ) -> Result<CurrentDensityTangent, ConstitutiveError> {
        Ok(IDENTITY_00 * -self.electrical_conductivity)
    }
}
