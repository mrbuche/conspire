//! Hyperviscoelastic solid constitutive models.
//!
//! ---
//!
//! Hyperviscoelastic solid constitutive models are defined by a Helmholtz free energy density and a viscous dissipation function.
//!
//! ```math
//! \mathbf{P}:\dot{\mathbf{F}} - \dot{a}(\mathbf{F}) \geq 0
//! ```
//! The second law of thermodynamics is satisfied by ensuring that the viscous dissipation function is convex and vanishes at zero rate,
//!
//! ```math
//! \frac{\partial\psi}{\partial\dot{\mathbf{F}}}:\dot{\mathbf{F}} - \psi \geq 0 \quad\text{and}\quad \psi(\mathbf{F},\mathbf{0}) = 0
//! ```
//! and by minimizing the dissipation potential less the stress power with respect to the deformation gradient rate, yielding the stress.
//!
//! ```math
//! \mathbf{P} = \frac{\partial\phi}{\partial\dot{\mathbf{F}}}
//! ```
//! Consequently, the rate tangent stiffness associated with the first Piola-Kirchhoff stress is symmetric for these constitutive models.
//!
//! ```math
//! \mathcal{U}_{iJkL} = \mathcal{U}_{kLiJ}
//! ```

#[cfg(test)]
pub mod test;

mod canonical;

#[cfg(feature = "autodiff")]
pub mod autodiff;

use super::{elastic_hyperviscous::ElasticHyperviscous, *};
use crate::{math::Quantity, units::EnergyDensity};

/// Required methods for hyperviscoelastic solid constitutive models.
pub trait Hyperviscoelastic
where
    Self: ElasticHyperviscous,
{
    /// Calculates and returns the Helmholtz free energy density.
    ///
    /// ```math
    /// a = a(\mathbf{F})
    /// ```
    fn helmholtz_free_energy_density(
        &self,
        deformation_gradient: &DeformationGradient,
    ) -> Result<Quantity<EnergyDensity>, ConstitutiveError>;
}
