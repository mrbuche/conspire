#[cfg(test)]
mod test;

use crate::{
    constitutive::{
        ConstitutiveError,
        solid::{FIVE_THIRDS, Solid, TWO_THIRDS, elastic::Elastic, hyperelastic::Hyperelastic},
    },
    math::{IDENTITY, Quantity, Rank2, TensorRank4},
    mechanics::{CauchyStress, CauchyTangentStiffness, Deformation, DeformationGradient},
    units::{EnergyDensity, Stress},
};

/// The Mooney-Rivlin hyperelastic solid constitutive model.[^1]<sup>,</sup>[^2]
///
/// **Parameters**
/// - The bulk modulus $`\kappa`$.
/// - The shear modulus $`\mu`$.
/// - The extra modulus $`\mu_m`$.
///
/// **External variables**
/// - The deformation gradient $`\mathbf{F}`$.
///
/// **Internal variables**
/// - None.
///
/// **Notes**
/// - The Mooney-Rivlin model reduces to the [Neo-Hookean model](super::NeoHookean) when $`\mu_m\to 0`$.
///
/// [^1]: M. Mooney, [J. Appl. Phys. **11**, 582 (1940)](https://doi.org/10.1063/1.1712836).
/// [^2]: R.S. Rivlin, [Philos. Trans. R. Soc. London, Ser. A **241**, 379 (1948)](https://doi.org/10.1098/rsta.1948.0024).
#[derive(Clone, Debug)]
pub struct MooneyRivlin {
    /// The bulk modulus $`\kappa`$.
    pub bulk_modulus: Quantity<Stress>,
    /// The shear modulus $`\mu`$.
    pub shear_modulus: Quantity<Stress>,
    /// The extra modulus $`\mu_m`$.
    pub extra_modulus: Quantity<Stress>,
}

impl MooneyRivlin {
    /// Returns the extra modulus.
    pub fn extra_modulus(&self) -> Quantity<Stress> {
        self.extra_modulus
    }
}

impl Solid for MooneyRivlin {
    fn bulk_modulus(&self) -> Quantity<Stress> {
        self.bulk_modulus
    }
    fn shear_modulus(&self) -> Quantity<Stress> {
        self.shear_modulus
    }
}

impl Elastic for MooneyRivlin {
    /// ```math
    /// \boldsymbol{\sigma}(\mathbf{F}) = \frac{\mu - \mu_m}{J}\, {\mathbf{B}^* }' - \frac{\mu_m}{J}\left(\mathbf{B}^{* -1}\right)' + \frac{\kappa}{2}\left(J - \frac{1}{J}\right)\mathbf{1}
    /// ```
    fn cauchy_stress(
        &self,
        deformation_gradient: &DeformationGradient,
    ) -> Result<CauchyStress, ConstitutiveError> {
        let jacobian = self.jacobian(deformation_gradient)?;
        let isochoric_left_cauchy_green_deformation =
            deformation_gradient.left_cauchy_green() / jacobian.powf(TWO_THIRDS);
        Ok(((isochoric_left_cauchy_green_deformation.deviatoric()
            * (self.shear_modulus() - self.extra_modulus())
            - isochoric_left_cauchy_green_deformation
                .inverse()
                .deviatoric()
                * self.extra_modulus())
            + IDENTITY * (self.bulk_modulus() * 0.5 * (jacobian.powi(2) - 1.0)))
            / jacobian)
    }
    /// ```math
    /// \begin{aligned}
    /// \mathcal{T}_{ijkL}(\mathbf{F}) =
    /// \,& \frac{\mu-\mu_m}{J^{5/3}}\left(\delta_{ik}F_{jL} + \delta_{jk}F_{iL} - \frac{2}{3}\,\delta_{ij}F_{kL}- \frac{5}{3} \, B_{ij}'F_{kL}^{-T} \right)
    /// + \frac{\kappa}{2} \left(J + \frac{1}{J}\right)\delta_{ij}F_{kL}^{-T}
    /// \\
    /// &- \frac{\mu_m}{J}\left[ \frac{2}{3}\,B_{ij}^{* -1}F_{kL}^{-T} - B_{ik}^{* -1}F_{jL}^{-T} - B_{ik}^{* -1}F_{iL}^{-T} + \frac{2}{3}\,\delta_{ij}\left(B_{km}^{* -1}\right)'F_{mL}^{-T} - \left(B_{ij}^{* -1}\right)'F_{kL}^{-T} \right]
    /// \end{aligned}
    /// ```
    fn cauchy_tangent_stiffness(
        &self,
        deformation_gradient: &DeformationGradient,
    ) -> Result<CauchyTangentStiffness, ConstitutiveError> {
        let jacobian = self.jacobian(deformation_gradient)?;
        let inverse_transpose_deformation_gradient = deformation_gradient.inverse_transpose();
        let scaled_delta_shear_modulus =
            (self.shear_modulus() - self.extra_modulus()) / jacobian.powf(FIVE_THIRDS);
        let inverse_isochoric_left_cauchy_green_deformation =
            (deformation_gradient.left_cauchy_green() / jacobian.powf(TWO_THIRDS)).inverse();
        let deviatoric_inverse_isochoric_left_cauchy_green_deformation =
            inverse_isochoric_left_cauchy_green_deformation.deviatoric();
        let term_1 = TensorRank4::dyad_ij_kl(
            &inverse_isochoric_left_cauchy_green_deformation,
            &inverse_transpose_deformation_gradient,
        ) * TWO_THIRDS
            - TensorRank4::dyad_ik_jl(
                &inverse_isochoric_left_cauchy_green_deformation,
                &inverse_transpose_deformation_gradient,
            )
            - TensorRank4::dyad_il_jk(
                &inverse_transpose_deformation_gradient,
                &inverse_isochoric_left_cauchy_green_deformation,
            );
        let term_3 = TensorRank4::dyad_ij_kl(
            &deviatoric_inverse_isochoric_left_cauchy_green_deformation,
            &inverse_transpose_deformation_gradient,
        );
        let term_2 = TensorRank4::dyad_ij_kl(
            &IDENTITY,
            &((deviatoric_inverse_isochoric_left_cauchy_green_deformation * TWO_THIRDS)
                * &inverse_transpose_deformation_gradient),
        );
        Ok((TensorRank4::dyad_ik_jl(&IDENTITY, deformation_gradient)
            + TensorRank4::dyad_il_jk(deformation_gradient, &IDENTITY)
            - TensorRank4::dyad_ij_kl(&IDENTITY, deformation_gradient) * (TWO_THIRDS))
            * scaled_delta_shear_modulus
            + TensorRank4::dyad_ij_kl(
                &(IDENTITY * (0.5 * self.bulk_modulus() * (jacobian + 1.0 / jacobian))
                    - deformation_gradient.left_cauchy_green().deviatoric()
                        * (scaled_delta_shear_modulus * FIVE_THIRDS)),
                &inverse_transpose_deformation_gradient,
            )
            - (term_1 + term_2 - term_3) * self.extra_modulus() / jacobian)
    }
}

impl Hyperelastic for MooneyRivlin {
    /// ```math
    /// a(\mathbf{F}) = \frac{\mu - \mu_m}{2}\left[\mathrm{tr}(\mathbf{B}^* ) - 3\right] + \frac{\mu_m}{2}\left[I_2(\mathbf{B}^*) - 3\right] + \frac{\kappa}{2}\left[\frac{1}{2}\left(J^2 - 1\right) - \ln J\right]
    /// ```
    fn helmholtz_free_energy_density(
        &self,
        deformation_gradient: &DeformationGradient,
    ) -> Result<Quantity<EnergyDensity>, ConstitutiveError> {
        let jacobian = self.jacobian(deformation_gradient)?;
        let isochoric_left_cauchy_green_deformation =
            deformation_gradient.left_cauchy_green() / jacobian.powf(TWO_THIRDS);
        Ok(0.5
            * ((self.shear_modulus() - self.extra_modulus())
                * (isochoric_left_cauchy_green_deformation.trace() - 3.0)
                + self.extra_modulus()
                    * (isochoric_left_cauchy_green_deformation.second_invariant() - 3.0)
                + self.bulk_modulus() * (0.5 * (jacobian.powi(2) - 1.0) - jacobian.ln())))
    }
}
