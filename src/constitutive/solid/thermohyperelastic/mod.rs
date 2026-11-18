//! Thermohyperelastic solid constitutive models.
//!
//! ---
//!
//! Thermohyperelastic solid constitutive models are defined by a Helmholtz free energy density function of the deformation gradient and temperature.
//!
//! ```math
//! \mathbf{P}:\dot{\mathbf{F}} - \dot{a}(\mathbf{F},T) - s\dot{T} \geq 0
//! ```
//! Satisfying the second law of thermodynamics (here, equivalent to extremized or zero dissipation) yields a relation for the stress.
//!
//! ```math
//! \mathbf{P} = \frac{\partial a}{\partial\mathbf{F}}
//! ```
//! Consequently, the tangent stiffness associated with the first Piola-Kirchhoff stress is symmetric for these constitutive models.
//!
//! ```math
//! \mathcal{C}_{iJkL} = \mathcal{C}_{kLiJ}
//! ```

#[cfg(test)]
pub mod test;

mod neo_hookean;
mod saint_venant_kirchhoff;

pub use neo_hookean::NeoHookean;
pub use saint_venant_kirchhoff::SaintVenantKirchhoff;

use super::{thermoelastic::Thermoelastic, *};
use crate::math::Quantity;
use crate::units::{EnergyDensity, Temperature};

/// Required methods for thermohyperelastic solid constitutive models.
pub trait Thermohyperelastic
where
    Self: Thermoelastic,
{
    /// Calculates and returns the Helmholtz free energy density.
    ///
    /// ```math
    /// a = a(\mathbf{F},T)
    /// ```
    fn helmholtz_free_energy_density(
        &self,
        deformation_gradient: &DeformationGradient,
        temperature: Quantity<Temperature>,
    ) -> Result<Quantity<EnergyDensity>, ConstitutiveError>;
}
