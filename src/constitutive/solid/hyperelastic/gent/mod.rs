#[cfg(test)]
mod test;

use crate::{
    constitutive::{
        ConstitutiveError,
        solid::{Solid, TWO_THIRDS, elastic::Elastic, hyperelastic::Hyperelastic},
    },
    math::{IDENTITY, Quantity, Rank2, TensorRank4},
    mechanics::{CauchyStress, CauchyTangentStiffness, Deformation, DeformationGradient, Scalar},
    units::{EnergyDensity, Stress},
};

/// The Gent hyperelastic solid constitutive model.[^1]
///
/// **Parameters**
/// - The bulk modulus $`\kappa`$.
/// - The shear modulus $`\mu`$.
/// - The extensibility $`J_m`$.
///
/// **External variables**
/// - The deformation gradient $`\mathbf{F}`$.
///
/// **Internal variables**
/// - None.
///
/// **Notes**
/// - The Gent model reduces to the [Neo-Hookean model](super::NeoHookean) when $`J_m\to\infty`$.
///
/// [^1]: A.N. Gent, [Rubber Chem. Technol. **69**, 59 (1996)](https://doi.org/10.5254/1.3538357).
#[derive(Clone, Debug)]
pub struct Gent {
    /// The bulk modulus $`\kappa`$.
    pub bulk_modulus: Quantity<Stress>,
    /// The shear modulus $`\mu`$.
    pub shear_modulus: Quantity<Stress>,
    /// The extensibility $`J_m`$.
    pub extensibility: Scalar,
}

impl Gent {
    /// Returns the extensibility.
    pub fn extensibility(&self) -> Scalar {
        self.extensibility
    }
}

impl Solid for Gent {
    fn bulk_modulus(&self) -> Quantity<Stress> {
        self.bulk_modulus
    }
    fn shear_modulus(&self) -> Quantity<Stress> {
        self.shear_modulus
    }
}

impl Elastic for Gent {
    /// ```math
    /// \boldsymbol{\sigma}(\mathbf{F}) = \frac{J^{-1}\mu J_m {\mathbf{B}^* }'}{J_m - \mathrm{tr}(\mathbf{B}^* ) + 3} + \frac{\kappa}{2}\left(J - \frac{1}{J}\right)\mathbf{1}
    /// ```
    fn cauchy_stress(
        &self,
        deformation_gradient: &DeformationGradient,
    ) -> Result<CauchyStress, ConstitutiveError> {
        let jacobian = self.jacobian(deformation_gradient)?;
        let isochoric_left_cauchy_green_deformation =
            deformation_gradient.left_cauchy_green() / jacobian.powf(TWO_THIRDS);
        let (
            deviatoric_isochoric_left_cauchy_green_deformation,
            isochoric_left_cauchy_green_deformation_trace,
        ) = isochoric_left_cauchy_green_deformation.deviatoric_and_trace();
        let denominator =
            self.extensibility() - isochoric_left_cauchy_green_deformation_trace + 3.0;
        if denominator <= 0.0 {
            Err(ConstitutiveError::custom(
                "Maximum extensibility reached.",
                self,
            ))
        } else {
            Ok((deviatoric_isochoric_left_cauchy_green_deformation
                * self.shear_modulus()
                * self.extensibility()
                / jacobian)
                / denominator
                + IDENTITY * self.bulk_modulus() * 0.5 * (jacobian - 1.0 / jacobian))
        }
    }
    /// ```math
    /// \begin{aligned}
    /// \mathcal{T}_{ijkL}(\mathbf{F}) =
    /// \,& \frac{J^{-5/3}\mu J_m}{J_m - \mathrm{tr}(\mathbf{B}^* ) + 3}\Bigg[\delta_{ik}F_{jL} + \delta_{jk}F_{iL} - \frac{2}{3}\,\delta_{ij}F_{kL} + \frac{2{B_{ij}^* }' F_{kL}}{J_m - \mathrm{tr}(\mathbf{B}^* ) + 3}
    /// \\
    /// &- \left(\frac{5}{3} + \frac{2}{3}\frac{\mathrm{tr}(\mathbf{B}^* )}{J_m - \mathrm{tr}(\mathbf{B}^* ) + 3}\right) J^{2/3} {B_{ij}^* }' F_{kL}^{-T} \Bigg] + \frac{\kappa}{2} \left(J + \frac{1}{J}\right)\delta_{ij}F_{kL}^{-T}
    /// \end{aligned}
    /// ```
    fn cauchy_tangent_stiffness(
        &self,
        deformation_gradient: &DeformationGradient,
    ) -> Result<CauchyTangentStiffness, ConstitutiveError> {
        let jacobian = self.jacobian(deformation_gradient)?;
        let inverse_transpose_deformation_gradient = deformation_gradient.inverse_transpose();
        let isochoric_left_cauchy_green_deformation =
            deformation_gradient.left_cauchy_green() / jacobian.powf(TWO_THIRDS);
        let (
            deviatoric_isochoric_left_cauchy_green_deformation,
            isochoric_left_cauchy_green_deformation_trace,
        ) = isochoric_left_cauchy_green_deformation.deviatoric_and_trace();
        let denominator =
            self.extensibility() - isochoric_left_cauchy_green_deformation_trace + 3.0;
        if denominator <= 0.0 {
            Err(ConstitutiveError::custom(
                "Maximum extensibility reached.",
                self,
            ))
        } else {
            let prefactor = self.shear_modulus() * self.extensibility() / jacobian / denominator;
            Ok((TensorRank4::dyad_ik_jl(&IDENTITY, deformation_gradient)
                + TensorRank4::dyad_il_jk(deformation_gradient, &IDENTITY)
                - TensorRank4::dyad_ij_kl(&IDENTITY, deformation_gradient) * (TWO_THIRDS)
                + TensorRank4::dyad_ij_kl(
                    &deviatoric_isochoric_left_cauchy_green_deformation,
                    deformation_gradient,
                ) * (2.0 / denominator))
                * (prefactor / jacobian.powf(TWO_THIRDS))
                + TensorRank4::dyad_ij_kl(
                    &(IDENTITY * (0.5 * self.bulk_modulus() * (jacobian + 1.0 / jacobian))
                        - deviatoric_isochoric_left_cauchy_green_deformation
                            * prefactor
                            * ((5.0
                                + 2.0 * isochoric_left_cauchy_green_deformation_trace
                                    / denominator)
                                / 3.0)),
                    &inverse_transpose_deformation_gradient,
                ))
        }
    }
}

impl Hyperelastic for Gent {
    /// ```math
    /// a(\mathbf{F}) = -\frac{\mu J_m}{2}\,\ln\left[1 - \frac{\mathrm{tr}(\mathbf{B}^* ) - 3}{J_m}\right] + \frac{\kappa}{2}\left[\frac{1}{2}\left(J^2 - 1\right) - \ln J\right]
    /// ```
    fn helmholtz_free_energy_density(
        &self,
        deformation_gradient: &DeformationGradient,
    ) -> Result<Quantity<EnergyDensity>, ConstitutiveError> {
        let jacobian = self.jacobian(deformation_gradient)?;
        let factor = (deformation_gradient.left_cauchy_green().trace() / jacobian.powf(TWO_THIRDS)
            - 3.0)
            / self.extensibility();
        if factor >= 1.0 {
            Err(ConstitutiveError::custom(
                "Maximum extensibility reached.",
                self,
            ))
        } else {
            Ok(0.5
                * (-self.shear_modulus() * self.extensibility() * (1.0 - factor).ln()
                    + self.bulk_modulus() * (0.5 * (jacobian.powi(2) - 1.0) - jacobian.ln())))
        }
    }
}
