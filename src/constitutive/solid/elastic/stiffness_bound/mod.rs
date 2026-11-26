//! Bounds on how stiff elastic solid constitutive models are about a deformation.

#[cfg(test)]
mod test;

use super::Elastic;
use crate::{
    constitutive::ConstitutiveError,
    math::{Quantity, Scalar},
    mechanics::DeformationGradient,
    units::Stress,
};

pub(crate) const FRAC_1_SQRT_3: Scalar = 0.577_350_269_189_625_8; // nightly feature
pub(crate) const SQRT_2_3: Scalar = 0.816_496_580_927_726;

/// Elastic models with moduli that bound how stiff they are about a deformation.
///
/// With $`\boldsymbol{\mathcal{C}}`$ the tangent stiffness of the first Piola-Kirchhoff stress
/// and $`\mathbf{h}=\delta\mathbf{F}\cdot\mathbf{F}^{-1}`$, the moduli are such that for every
/// $`\delta\mathbf{F}`$
/// ```math
/// \delta\mathbf{F}:\boldsymbol{\mathcal{C}}:\delta\mathbf{F} \leq J\left[\hat\lambda\,\mathrm{tr}(\mathbf{h})^2 + 2\hat\mu\,\mathbf{h}:\mathbf{h}\right]
/// ```
/// where $`\hat\lambda`$ is not negative. They are the Lamé moduli in the reference
/// configuration, if the first of those is not negative.
pub trait StiffnessBound
where
    Self: Elastic,
{
    /// Calculates and returns the moduli $`(\hat\lambda,\hat\mu)`$.
    fn stiffness_bound(
        &self,
        deformation_gradient: &DeformationGradient,
    ) -> Result<(Quantity<Stress>, Quantity<Stress>), ConstitutiveError>;
}
