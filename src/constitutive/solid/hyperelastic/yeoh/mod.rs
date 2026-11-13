#[cfg(test)]
mod test;

use crate::{
    constitutive::{
        ConstitutiveError,
        solid::{FIVE_THIRDS, Solid, TWO_THIRDS, elastic::Elastic, hyperelastic::Hyperelastic},
    },
    math::{IDENTITY, Quantity, Rank2, TensorRank4},
    mechanics::{CauchyStress, CauchyTangentStiffness, Deformation, DeformationGradient, Scalar},
    units::{EnergyDensity, Modulus, Stress},
};

const SEVEN_THIRDS: Scalar = 7.0 / 3.0;

/// The Yeoh hyperelastic solid constitutive model.[^1]
///
/// **Parameters**
/// - The bulk modulus $`\kappa`$.
/// - The shear moduli $`\mu_n`$ for $`n=1\ldots N`$.
///
/// **External variables**
/// - The deformation gradient $`\mathbf{F}`$.
///
/// **Internal variables**
/// - None.
///
/// **Notes**
/// - The shear modulus is given by $`\mu=\mu_1`$.
/// - The Yeoh model reduces to the [Neo-Hookean model](super::NeoHookean) when $`N=1`$.
///
/// [^1]: O.H. Yeoh, [Rubber Chem. Technol. **66**, 754 (1993)](https://doi.org/10.5254/1.3538343).
#[derive(Clone, Debug)]
pub struct Yeoh {
    /// The bulk modulus $`\kappa`$.
    pub bulk_modulus: Quantity<Stress>,
    /// The shear moduli $`\mu_n`$.
    pub shear_moduli: Vec<Quantity<Modulus>>,
}

impl Solid for Yeoh {
    fn bulk_modulus(&self) -> Quantity<Stress> {
        self.bulk_modulus
    }
    fn shear_modulus(&self) -> Quantity<Stress> {
        self.shear_moduli.first().copied().unwrap_or_default()
    }
}

impl Elastic for Yeoh {
    /// ```math
    /// \boldsymbol{\sigma}(\mathbf{F}) = \sum_{n=1}^N \frac{n\mu_n}{J}\left[\mathrm{tr}(\mathbf{B}^* ) - 3\right]^{n-1}\,{\mathbf{B}^*}' + \frac{\kappa}{2}\left(J - \frac{1}{J}\right)\mathbf{1}
    /// ```
    fn cauchy_stress(
        &self,
        deformation_gradient: &DeformationGradient,
    ) -> Result<CauchyStress, ConstitutiveError> {
        let jacobian = self.jacobian(deformation_gradient)?;
        let (deviatoric_left_cauchy_green_deformation, left_cauchy_green_deformation_trace) =
            deformation_gradient
                .left_cauchy_green()
                .deviatoric_and_trace();
        let scalar_term = left_cauchy_green_deformation_trace / jacobian.powf(TWO_THIRDS) - 3.0;
        Ok(deviatoric_left_cauchy_green_deformation
            * self
                .shear_moduli
                .iter()
                .enumerate()
                .map(|(n, &modulus)| modulus * (((n as Scalar) + 1.0) * scalar_term.powi(n as i32)))
                .sum::<Quantity<Modulus>>()
            / jacobian.powf(FIVE_THIRDS)
            + IDENTITY * self.bulk_modulus() * 0.5 * (jacobian - 1.0 / jacobian))
    }
    /// ```math
    /// \begin{aligned}
    /// \mathcal{T}_{ijkL}(\mathbf{F}) =
    /// \,& \sum_{n=1}^N \frac{n\mu_n}{J^{5/3}}\left[\mathrm{tr}(\mathbf{B}^* ) - 3\right]^{n-1}\left(\delta_{ik}F_{jL} + \delta_{jk}F_{iL} - \frac{2}{3}\,\delta_{ij}F_{kL}- \frac{5}{3} \, B_{ij}'F_{kL}^{-T} \right)
    /// \\
    /// &+ \sum_{n=2}^N \frac{2n(n-1)\mu_n}{J^{7/3}}\left[\mathrm{tr}(\mathbf{B}^* ) - 3\right]^{n-2}B_{ij}'B_{km}'F_{mL}^{-T} + \frac{\kappa}{2} \left(J + \frac{1}{J}\right)\delta_{ij}F_{kL}^{-T}
    /// \end{aligned}
    /// ```
    fn cauchy_tangent_stiffness(
        &self,
        deformation_gradient: &DeformationGradient,
    ) -> Result<CauchyTangentStiffness, ConstitutiveError> {
        let jacobian = self.jacobian(deformation_gradient)?;
        let inverse_transpose_deformation_gradient = deformation_gradient.inverse_transpose();
        let left_cauchy_green_deformation = deformation_gradient.left_cauchy_green();
        let scalar_term = left_cauchy_green_deformation.trace() / jacobian.powf(TWO_THIRDS) - 3.0;
        let scaled_modulus = self
            .shear_moduli
            .iter()
            .enumerate()
            .map(|(n, &modulus)| modulus * (((n as Scalar) + 1.0) * scalar_term.powi(n as i32)))
            .sum::<Quantity<Modulus>>()
            / jacobian.powf(FIVE_THIRDS);
        let deviatoric_left_cauchy_green_deformation = left_cauchy_green_deformation.deviatoric();
        let last_term = TensorRank4::dyad_ij_kl(
            &deviatoric_left_cauchy_green_deformation,
            &((left_cauchy_green_deformation.deviatoric()
                * &inverse_transpose_deformation_gradient)
                * (self
                    .shear_moduli
                    .iter()
                    .enumerate()
                    .skip(1)
                    .map(|(n, &modulus)| {
                        modulus
                            * (2.0
                                * ((n as Scalar) + 1.0)
                                * (n as Scalar)
                                * scalar_term.powi(n as i32 - 1))
                    })
                    .sum::<Quantity<Modulus>>()
                    / jacobian.powf(SEVEN_THIRDS))),
        );
        Ok((TensorRank4::dyad_ik_jl(&IDENTITY, deformation_gradient)
            + TensorRank4::dyad_il_jk(deformation_gradient, &IDENTITY)
            - TensorRank4::dyad_ij_kl(&IDENTITY, deformation_gradient) * (TWO_THIRDS))
            * scaled_modulus
            + TensorRank4::dyad_ij_kl(
                &(IDENTITY * (self.bulk_modulus() * 0.5 * (jacobian + 1.0 / jacobian))
                    - deviatoric_left_cauchy_green_deformation * (scaled_modulus * FIVE_THIRDS)),
                &inverse_transpose_deformation_gradient,
            )
            + last_term)
    }
}

impl Hyperelastic for Yeoh {
    /// ```math
    /// a(\mathbf{F}) = \sum_{n=1}^N \frac{\mu_n}{2}\big[\mathrm{tr}(\mathbf{B}^* ) - 3\big]^n + \frac{\kappa}{2}\left[\frac{1}{2}\left(J^2 - 1\right) - \ln J\right]
    /// ```
    fn helmholtz_free_energy_density(
        &self,
        deformation_gradient: &DeformationGradient,
    ) -> Result<Quantity<EnergyDensity>, ConstitutiveError> {
        let jacobian = self.jacobian(deformation_gradient)?;
        let scalar_term =
            deformation_gradient.left_cauchy_green().trace() / jacobian.powf(TWO_THIRDS) - 3.0;
        Ok(0.5
            * (self
                .shear_moduli
                .iter()
                .enumerate()
                .map(|(n, &modulus)| modulus * scalar_term.powi((n + 1) as i32))
                .sum::<Quantity<Modulus>>()
                + self.bulk_modulus() * (0.5 * (jacobian.powi(2) - 1.0) - jacobian.ln())))
    }
}
