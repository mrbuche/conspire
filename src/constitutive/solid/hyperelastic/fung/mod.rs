#[cfg(test)]
mod test;

use crate::{
    constitutive::{
        ConstitutiveError,
        solid::{FIVE_THIRDS, Solid, TWO_THIRDS, elastic::Elastic, hyperelastic::Hyperelastic},
    },
    math::{IDENTITY, Quantity, Rank2, TensorRank4},
    mechanics::{CauchyStress, CauchyTangentStiffness, Deformation, DeformationGradient, Scalar},
    units::{EnergyDensity, Stress},
};

/// The Fung hyperelastic solid constitutive model.[^1]
///
/// **Parameters**
/// - The bulk modulus $`\kappa`$.
/// - The shear modulus $`\mu`$.
/// - The extra modulus $`\mu_m`$.
/// - The exponent $`c`$.
///
/// **External variables**
/// - The deformation gradient $`\mathbf{F}`$.
///
/// **Internal variables**
/// - None.
///
/// **Notes**
/// - The Fung model reduces to the [Neo-Hookean model](super::NeoHookean) when $`\mu_m\to 0`$ or $`c\to 0`$.
///
/// [^1]: Y.C. Fung, [Am. J. Physiol. **213**, 1532 (1967)](https://doi.org/10.1152/ajplegacy.1967.213.6.1532).
#[derive(Clone, Debug)]
pub struct Fung {
    /// The bulk modulus $`\kappa`$.
    pub bulk_modulus: Quantity<Stress>,
    /// The shear modulus $`\mu`$.
    pub shear_modulus: Quantity<Stress>,
    /// The extra modulus $`\mu_m`$.
    pub extra_modulus: Quantity<Stress>,
    /// The exponent $`c`$.
    pub exponent: Scalar,
}

impl Fung {
    /// Returns the extra modulus.
    pub fn extra_modulus(&self) -> Quantity<Stress> {
        self.extra_modulus
    }
    /// Returns the exponent.
    pub fn exponent(&self) -> Scalar {
        self.exponent
    }
}

impl Solid for Fung {
    fn bulk_modulus(&self) -> Quantity<Stress> {
        self.bulk_modulus
    }
    fn shear_modulus(&self) -> Quantity<Stress> {
        self.shear_modulus
    }
}

impl Elastic for Fung {
    /// ```math
    /// \boldsymbol{\sigma}(\mathbf{F}) = \frac{1}{J}\left[\mu + \mu_m\left(e^{c[\mathrm{tr}(\mathbf{B}^* ) - 3]} - 1\right)\right]{\mathbf{B}^* }' + \frac{\kappa}{2}\left(J - \frac{1}{J}\right)\mathbf{1}
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
        Ok(deviatoric_isochoric_left_cauchy_green_deformation
            * ((self.shear_modulus()
                + self.extra_modulus()
                    * ((self.exponent() * (isochoric_left_cauchy_green_deformation_trace - 3.0))
                        .exp()
                        - 1.0))
                / jacobian)
            + IDENTITY * self.bulk_modulus() * 0.5 * (jacobian - 1.0 / jacobian))
    }
    /// ```math
    /// \begin{aligned}
    /// \mathcal{T}_{ijkL}(\mathbf{F}) =
    /// \,& \frac{1}{J^{5/3}}\left[\mu + \mu_m\left(e^{c[\mathrm{tr}(\mathbf{B}^* ) - 3]} - 1\right)\right]\left(\delta_{ik}F_{jL} + \delta_{jk}F_{iL} - \frac{2}{3}\,\delta_{ij}F_{kL} - \frac{5}{3} \, B_{ij}'F_{kL}^{-T} \right)
    /// \\
    /// &+ \frac{2c\mu_m}{J^{7/3}}\,e^{c[\mathrm{tr}(\mathbf{B}^* ) - 3]}B_{ij}'B_{km}'F_{mL}^{-T} + \frac{\kappa}{2} \left(J + \frac{1}{J}\right)\delta_{ij}F_{kL}^{-T}
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
        let exponential =
            (self.exponent() * (isochoric_left_cauchy_green_deformation_trace - 3.0)).exp();
        let scaled_shear_modulus_0 = (self.shear_modulus()
            + self.extra_modulus() * (exponential - 1.0))
            / jacobian.powf(FIVE_THIRDS);
        Ok((TensorRank4::dyad_ik_jl(&IDENTITY, deformation_gradient)
            + TensorRank4::dyad_il_jk(deformation_gradient, &IDENTITY)
            - TensorRank4::dyad_ij_kl(&IDENTITY, deformation_gradient) * (TWO_THIRDS))
            * scaled_shear_modulus_0
            + TensorRank4::dyad_ij_kl(
                &deviatoric_isochoric_left_cauchy_green_deformation,
                &((&deviatoric_isochoric_left_cauchy_green_deformation
                    * &inverse_transpose_deformation_gradient)
                    * (2.0 * self.exponent() * self.extra_modulus() * exponential / jacobian)),
            )
            + TensorRank4::dyad_ij_kl(
                &(IDENTITY * (0.5 * self.bulk_modulus() * (jacobian + 1.0 / jacobian))
                    - deformation_gradient.left_cauchy_green().deviatoric()
                        * (scaled_shear_modulus_0 * FIVE_THIRDS)),
                &inverse_transpose_deformation_gradient,
            ))
    }
}

impl Hyperelastic for Fung {
    /// ```math
    /// a(\mathbf{F}) = \frac{\mu - \mu_m}{2}\left[\mathrm{tr}(\mathbf{B}^* ) - 3\right] + \frac{\mu_m}{2c}\left(e^{c[\mathrm{tr}(\mathbf{B}^* ) - 3]} - 1\right)
    /// ```
    fn helmholtz_free_energy_density(
        &self,
        deformation_gradient: &DeformationGradient,
    ) -> Result<Quantity<EnergyDensity>, ConstitutiveError> {
        let jacobian = self.jacobian(deformation_gradient)?;
        let scalar_term =
            deformation_gradient.left_cauchy_green().trace() / jacobian.powf(TWO_THIRDS) - 3.0;
        Ok(0.5
            * ((self.shear_modulus() - self.extra_modulus()) * scalar_term
                + self.extra_modulus() / self.exponent()
                    * ((self.exponent() * scalar_term).exp() - 1.0)
                + self.bulk_modulus() * (0.5 * (jacobian.powi(2) - 1.0) - jacobian.ln())))
    }
}
