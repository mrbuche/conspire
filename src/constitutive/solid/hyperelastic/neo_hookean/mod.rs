#[cfg(test)]
mod test;

use crate::{
    constitutive::{
        ConstitutiveError, FromParameters,
        solid::{
            FIVE_THIRDS, Solid, TWO_THIRDS,
            elastic::{Elastic, FRAC_1_SQRT_3, SQRT_2_3, StiffnessBound},
            hyperelastic::Hyperelastic,
        },
    },
    math::{IDENTITY, Quantity, Rank2, Tensor, TensorRank4},
    mechanics::{CauchyStress, CauchyTangentStiffness, Deformation, DeformationGradient},
    units::{EnergyDensity, Stress},
};

/// The Neo-Hookean hyperelastic solid constitutive model.[^1]
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
/// [^1]: R.S. Rivlin, [Philos. Trans. R. Soc. London, Ser. A **240**, 459 (1948)](https://doi.org/10.1098/rsta.1948.0002).
#[derive(Clone, Debug)]
pub struct NeoHookean {
    /// The bulk modulus $`\kappa`$.
    pub bulk_modulus: Quantity<Stress>,
    /// The shear modulus $`\mu`$.
    pub shear_modulus: Quantity<Stress>,
}

impl Solid for NeoHookean {
    fn bulk_modulus(&self) -> Quantity<Stress> {
        self.bulk_modulus
    }
    fn shear_modulus(&self) -> Quantity<Stress> {
        self.shear_modulus
    }
}

impl FromParameters for NeoHookean {
    const NAME: &'static str = "neo_hookean";
    fn parameters(&self) -> Vec<f64> {
        vec![self.shear_modulus.value(), self.bulk_modulus.value()]
    }
    fn from_parameters(parameters: &[f64]) -> Self {
        Self {
            shear_modulus: Stress::pascals(parameters[0]),
            bulk_modulus: Stress::pascals(parameters[1]),
        }
    }
}

impl Elastic for NeoHookean {
    /// ```math
    /// \boldsymbol{\sigma}(\mathbf{F}) = \frac{\mu}{J}\,{\mathbf{B}^*}' + \frac{\kappa}{2}\left(J - \frac{1}{J}\right)\mathbf{1}
    /// ```
    fn cauchy_stress(
        &self,
        deformation_gradient: &DeformationGradient,
    ) -> Result<CauchyStress, ConstitutiveError> {
        let jacobian = self.jacobian(deformation_gradient)?;
        Ok(
            deformation_gradient.left_cauchy_green().deviatoric() / jacobian.powf(FIVE_THIRDS)
                * self.shear_modulus()
                + IDENTITY * self.bulk_modulus() * 0.5 * (jacobian - 1.0 / jacobian),
        )
    }
    /// ```math
    /// \mathcal{T}_{ijkL}(\mathbf{F}) = \frac{\mu}{J^{5/3}}\left(\delta_{ik}F_{jL} + \delta_{jk}F_{iL} - \frac{2}{3}\,\delta_{ij}F_{kL} - \frac{5}{3} \, B_{ij}'F_{kL}^{-T} \right) + \frac{\kappa}{2} \left(J + \frac{1}{J}\right)\delta_{ij}F_{kL}^{-T}
    /// ```
    fn cauchy_tangent_stiffness(
        &self,
        deformation_gradient: &DeformationGradient,
    ) -> Result<CauchyTangentStiffness, ConstitutiveError> {
        let jacobian = self.jacobian(deformation_gradient)?;
        let inverse_transpose_deformation_gradient = deformation_gradient.inverse_transpose();
        let scaled_shear_modulus = self.shear_modulus() / jacobian.powf(FIVE_THIRDS);
        Ok((TensorRank4::dyad_ik_jl(&IDENTITY, deformation_gradient)
            + TensorRank4::dyad_il_jk(deformation_gradient, &IDENTITY)
            - TensorRank4::dyad_ij_kl(&IDENTITY, deformation_gradient) * (TWO_THIRDS))
            * scaled_shear_modulus
            + TensorRank4::dyad_ij_kl(
                &(IDENTITY * (self.bulk_modulus() * 0.5 * (jacobian + 1.0 / jacobian))
                    - deformation_gradient.left_cauchy_green().deviatoric()
                        * (scaled_shear_modulus * FIVE_THIRDS)),
                &inverse_transpose_deformation_gradient,
            ))
    }
}

impl StiffnessBound for NeoHookean {
    /// ```math
    /// \hat\lambda = \max\left[0,\,\kappa J - \frac{2\mu}{3J}\left(a - \frac{b}{\sqrt{3}}\right)\right]
    /// ```
    /// ```math
    /// \hat\mu = \frac{\mu}{2J}\left[2a + \left(\sqrt{\tfrac{2}{3}} + \tfrac{2}{\sqrt{3}}\right)b\right] + \frac{\kappa}{4}\left|J - \frac{1}{J}\right|
    /// ```
    /// where $`a=\mathrm{tr}(\mathbf{B}^*)/3`$ and $`b=|{\mathbf{B}^*}'|`$.
    fn stiffness_bound(
        &self,
        deformation_gradient: &DeformationGradient,
    ) -> Result<(Quantity<Stress>, Quantity<Stress>), ConstitutiveError> {
        let jacobian = self.jacobian(deformation_gradient)?;
        let (deviator, trace) = (deformation_gradient.left_cauchy_green()
            / jacobian.powf(TWO_THIRDS))
        .deviatoric_and_trace();
        let (mean, spread) = (trace.value() / 3.0, deviator.norm().value());
        let shear_modulus = self.shear_modulus() / jacobian;
        Ok((
            (self.bulk_modulus() * jacobian
                - shear_modulus * (TWO_THIRDS * (mean - spread * FRAC_1_SQRT_3)))
                .max(Quantity::default()),
            shear_modulus * (mean + (0.5 * SQRT_2_3 + FRAC_1_SQRT_3) * spread)
                + self.bulk_modulus() * (0.25 * (jacobian - 1.0 / jacobian).abs()),
        ))
    }
}

impl Hyperelastic for NeoHookean {
    /// ```math
    /// a(\mathbf{F}) = \frac{\mu}{2}\left[\mathrm{tr}(\mathbf{B}^*) - 3\right] + \frac{\kappa}{2}\left[\frac{1}{2}\left(J^2 - 1\right) - \ln J\right]
    /// ```
    fn helmholtz_free_energy_density(
        &self,
        deformation_gradient: &DeformationGradient,
    ) -> Result<Quantity<EnergyDensity>, ConstitutiveError> {
        let jacobian = self.jacobian(deformation_gradient)?;
        Ok(0.5
            * (self.shear_modulus()
                * (deformation_gradient.left_cauchy_green().trace() / jacobian.powf(TWO_THIRDS)
                    - 3.0)
                + self.bulk_modulus() * (0.5 * (jacobian.powi(2) - 1.0) - jacobian.ln())))
    }
}
