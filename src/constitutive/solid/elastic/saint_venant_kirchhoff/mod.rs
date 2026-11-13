#[cfg(test)]
mod test;

use crate::{
    constitutive::{
        ConstitutiveError,
        solid::{Solid, TWO_THIRDS, elastic::Elastic},
    },
    math::{IDENTITY, Quantity, Rank2, TensorRank4},
    mechanics::{CauchyStress, CauchyTangentStiffness, Deformation, DeformationGradient},
    units::Stress,
};

/// The Saint Venant-Kirchhoff elastic solid constitutive model.
///
/// **Parameters**
/// - The bulk modulus $`\kappa`$.
/// - The shear modulus $`\mu`$.
///
/// **External variables**
/// - The deformation gradient $`\mathbf{F}`$.
///
/// **Internal variables**
/// - None.
///
/// **Notes**
/// - The Green-Saint Venant strain measure is given by $`\mathbf{e}=\tfrac{1}{2}(\mathbf{B}-\mathbf{1})`$.
#[derive(Clone, Debug)]
pub struct SaintVenantKirchhoff {
    /// The bulk modulus $`\kappa`$.
    pub bulk_modulus: Quantity<Stress>,
    /// The shear modulus $`\mu`$.
    pub shear_modulus: Quantity<Stress>,
}

impl Solid for SaintVenantKirchhoff {
    fn bulk_modulus(&self) -> Quantity<Stress> {
        self.bulk_modulus
    }
    fn shear_modulus(&self) -> Quantity<Stress> {
        self.shear_modulus
    }
}

impl Elastic for SaintVenantKirchhoff {
    /// ```math
    /// \boldsymbol{\sigma}(\mathbf{F}) = \frac{2\mu}{J}\,\mathbf{e}' + \frac{\kappa}{J}\,\mathrm{tr}(\mathbf{e})\mathbf{1}
    /// ```
    fn cauchy_stress(
        &self,
        deformation_gradient: &DeformationGradient,
    ) -> Result<CauchyStress, ConstitutiveError> {
        let jacobian = self.jacobian(deformation_gradient)?;
        let (deviatoric_strain, strain_trace) =
            ((deformation_gradient.left_cauchy_green() - IDENTITY) * 0.5).deviatoric_and_trace();
        Ok(deviatoric_strain * (2.0 * self.shear_modulus() / jacobian)
            + IDENTITY * (self.bulk_modulus() * strain_trace / jacobian))
    }
    /// ```math
    /// \mathcal{T}_{ijkL}(\mathbf{F}) = \frac{\mu}{J}\left[\delta_{jk}F_{iL} + \delta_{ik}F_{jL} - 2e_{ij}'F_{kL}^{-T}\right] + \frac{\lambda}{J}\,\delta_{ij}F_{kL} - \frac{\kappa}{J}\,\mathrm{tr}(\mathbf{e})\delta_{ij}F_{kL}^{-T}
    /// ```
    fn cauchy_tangent_stiffness(
        &self,
        deformation_gradient: &DeformationGradient,
    ) -> Result<CauchyTangentStiffness, ConstitutiveError> {
        let jacobian = self.jacobian(deformation_gradient)?;
        let inverse_transpose_deformation_gradient = deformation_gradient.inverse_transpose();
        let scaled_deformation_gradient = deformation_gradient * (self.shear_modulus() / jacobian);
        let (deviatoric_strain, strain_trace) =
            ((deformation_gradient.left_cauchy_green() - IDENTITY) * 0.5).deviatoric_and_trace();
        Ok(
            (TensorRank4::dyad_il_jk(&scaled_deformation_gradient, &IDENTITY)
                + TensorRank4::dyad_ik_jl(&IDENTITY, &scaled_deformation_gradient))
                + TensorRank4::dyad_ij_kl(
                    &IDENTITY,
                    &(deformation_gradient
                        * ((self.bulk_modulus() - self.shear_modulus() * TWO_THIRDS) / jacobian)),
                )
                - TensorRank4::dyad_ij_kl(
                    &(deviatoric_strain * (2.0 * self.shear_modulus() / jacobian)
                        + IDENTITY * (self.bulk_modulus() * strain_trace / jacobian)),
                    &inverse_transpose_deformation_gradient,
                ),
        )
    }
}
