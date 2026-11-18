#[cfg(test)]
mod test;

use crate::{
    constitutive::{ConstitutiveError, diffusion::Diffusion, electrical::Electrical},
    math::{IDENTITY_00, Quantity, Tensor},
    mechanics::{ElectricDisplacement, ElectricDisplacementTangent, ElectricField},
    units::{Charge, EnergyDensity, Permittivity},
};

/// The linear dielectric constitutive model.
///
/// **Parameters**
/// - The permittivity $`\varepsilon`$.
///
/// **External variables**
/// - The potential gradient $`\nabla\phi`$.
///
/// **Internal variables**
/// - None.
#[derive(Clone, Debug)]
pub struct Linear {
    /// The permittivity $`\varepsilon`$.
    pub permittivity: Quantity<Permittivity>,
}

impl Electrical for Linear {}

impl Diffusion for Linear {
    type Balance = Charge;
    /// ```math
    /// u(\nabla\phi) = \frac{1}{2}\varepsilon\nabla\phi\cdot\nabla\phi
    /// ```
    fn potential(
        &self,
        potential_gradient: &ElectricField,
    ) -> Result<Quantity<EnergyDensity>, ConstitutiveError> {
        Ok(potential_gradient.norm_squared() * self.permittivity * 0.5)
    }
    /// ```math
    /// \mathbf{D}(\nabla\phi) = -\varepsilon\nabla\phi
    /// ```
    fn flux(
        &self,
        potential_gradient: &ElectricField,
    ) -> Result<ElectricDisplacement, ConstitutiveError> {
        Ok(potential_gradient * -self.permittivity)
    }
    /// ```math
    /// \frac{\partial\mathbf{D}}{\partial\nabla\phi} = -\varepsilon\mathbf{I}
    /// ```
    fn tangent(
        &self,
        _potential_gradient: &ElectricField,
    ) -> Result<ElectricDisplacementTangent, ConstitutiveError> {
        Ok(IDENTITY_00 * -self.permittivity)
    }
}
