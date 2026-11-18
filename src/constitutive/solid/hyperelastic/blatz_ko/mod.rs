#[cfg(test)]
mod test;

use crate::{
    constitutive::{
        ConstitutiveError,
        solid::{Solid, elastic::Elastic, hyperelastic::Hyperelastic},
    },
    math::{IDENTITY, Quantity, Rank2, TensorRank4},
    mechanics::{CauchyStress, CauchyTangentStiffness, Deformation, DeformationGradient, Scalar},
    units::{EnergyDensity, Stress},
};

/// The Blatz-Ko hyperelastic solid constitutive model.[^1]
///
/// **Parameters**
/// - The bulk modulus $`\kappa`$.
/// - The shear modulus $`\mu`$.
/// - The mixing parameter $`f`$.
///
/// **External variables**
/// - The deformation gradient $`\mathbf{F}`$.
///
/// **Internal variables**
/// - None.
///
/// **Notes**
/// - The parameter $`n = \kappa/2\mu - 1/3`$ is determined by the bulk and shear moduli.
/// - This model reduces to the rubber case when $`f\to 1`$ and to the foam case when $`f\to 0`$.
/// - In the tangent stiffness, $`K = (1-f)J^{2n} - fJ^{-2n}`$ and $`K' = 2n\left[(1-f)J^{2n} + fJ^{-2n}\right]`$.
///
/// [^1]: P.J. Blatz and W.L. Ko, [Trans. Soc. Rheol. **6**, 223 (1962)](https://doi.org/10.1122/1.548937).
#[derive(Clone, Debug)]
pub struct BlatzKo {
    /// The bulk modulus $`\kappa`$.
    pub bulk_modulus: Quantity<Stress>,
    /// The shear modulus $`\mu`$.
    pub shear_modulus: Quantity<Stress>,
    /// The mixing parameter $`f`$.
    pub mixing_parameter: Scalar,
}

impl BlatzKo {
    /// Returns the mixing parameter.
    pub fn mixing_parameter(&self) -> Scalar {
        self.mixing_parameter
    }
    fn n(&self) -> Scalar {
        0.5 * self.bulk_modulus().value() / self.shear_modulus().value() - 1.0 / 3.0
    }
}

impl Solid for BlatzKo {
    fn bulk_modulus(&self) -> Quantity<Stress> {
        self.bulk_modulus
    }
    fn shear_modulus(&self) -> Quantity<Stress> {
        self.shear_modulus
    }
}

impl Elastic for BlatzKo {
    /// ```math
    /// \boldsymbol{\sigma}(\mathbf{F}) = \frac{\mu}{J}\Big\{f\mathbf{B} - (1 - f)\mathbf{B}^{-1} + \left[(1-f)J^{2n} - fJ^{-2n}\right]\mathbf{1}\Big\}
    /// ```
    fn cauchy_stress(
        &self,
        deformation_gradient: &DeformationGradient,
    ) -> Result<CauchyStress, ConstitutiveError> {
        let jacobian = self.jacobian(deformation_gradient)?;
        let n = self.n();
        let f = self.mixing_parameter();
        let left_cauchy_green_deformation = deformation_gradient.left_cauchy_green();
        let inverse_left_cauchy_green_deformation = left_cauchy_green_deformation.inverse();
        let k = (1.0 - f) * jacobian.powf(2.0 * n) - f * jacobian.powf(-2.0 * n);
        Ok(
            (left_cauchy_green_deformation * f - inverse_left_cauchy_green_deformation * (1.0 - f)
                + IDENTITY * k)
                * (self.shear_modulus() / jacobian),
        )
    }
    /// ```math
    /// \begin{aligned}
    /// \mathcal{T}_{ijkL}(\mathbf{F}) =
    /// \,& \frac{\mu f}{J}\Big[\delta_{ik}F_{jL} + \delta_{jk}F_{iL} - B_{ij}F_{kL}^{-T}\Big]
    /// \\
    /// &+ \frac{\mu(1-f)}{J}\Big[B_{ik}^{-1}F_{jL}^{-T} + F_{iL}^{-T}B_{jk}^{-1} + B_{ij}^{-1}F_{kL}^{-T}\Big]
    /// \\
    /// &+ \frac{\mu}{J}\left(K' - K\right)\delta_{ij}F_{kL}^{-T}
    /// \end{aligned}
    /// ```
    fn cauchy_tangent_stiffness(
        &self,
        deformation_gradient: &DeformationGradient,
    ) -> Result<CauchyTangentStiffness, ConstitutiveError> {
        let jacobian = self.jacobian(deformation_gradient)?;
        let inverse_transpose_deformation_gradient = deformation_gradient.inverse_transpose();
        let n = self.n();
        let f = self.mixing_parameter();
        let left_cauchy_green_deformation = deformation_gradient.left_cauchy_green();
        let inverse_left_cauchy_green_deformation = left_cauchy_green_deformation.inverse();
        let jacobian_pos = jacobian.powf(2.0 * n);
        let jacobian_neg = jacobian.powf(-2.0 * n);
        let k = (1.0 - f) * jacobian_pos - f * jacobian_neg;
        let k_prime = 2.0 * n * ((1.0 - f) * jacobian_pos + f * jacobian_neg);
        let shear_modulus_over_jacobian = self.shear_modulus() / jacobian;
        Ok(((TensorRank4::dyad_ik_jl(&IDENTITY, deformation_gradient)
            + TensorRank4::dyad_il_jk(deformation_gradient, &IDENTITY)
            - TensorRank4::dyad_ij_kl(
                &left_cauchy_green_deformation,
                &inverse_transpose_deformation_gradient,
            ))
            * f
            + (TensorRank4::dyad_ik_jl(
                &inverse_left_cauchy_green_deformation,
                &inverse_transpose_deformation_gradient,
            ) + TensorRank4::dyad_il_jk(
                &inverse_transpose_deformation_gradient,
                &inverse_left_cauchy_green_deformation,
            ) + TensorRank4::dyad_ij_kl(
                &inverse_left_cauchy_green_deformation,
                &inverse_transpose_deformation_gradient,
            )) * (1.0 - f)
            + TensorRank4::dyad_ij_kl(&IDENTITY, &inverse_transpose_deformation_gradient)
                * (k_prime - k))
            * shear_modulus_over_jacobian)
    }
}

impl Hyperelastic for BlatzKo {
    /// ```math
    /// a(\mathbf{F}) = \frac{\mu}{2}\left\{f\left[I_1 - 3 + \frac{1}{n}\left(I_3^{-n} - 1\right)\right] + (1-f)\left[\frac{I_2}{I_3} - 3 + \frac{1}{n}\left(I_3^n - 1\right)\right]\right\}
    /// ```
    fn helmholtz_free_energy_density(
        &self,
        deformation_gradient: &DeformationGradient,
    ) -> Result<Quantity<EnergyDensity>, ConstitutiveError> {
        let jacobian = self.jacobian(deformation_gradient)?;
        let n = self.n();
        let f = self.mixing_parameter();
        let left_cauchy_green_deformation = deformation_gradient.left_cauchy_green();
        let first_invariant = left_cauchy_green_deformation.trace();
        let second_invariant = left_cauchy_green_deformation.second_invariant();
        let third_invariant = jacobian.powi(2);
        Ok(0.5
            * self.shear_modulus()
            * (f * (first_invariant - 3.0 + (third_invariant.powf(-n) - 1.0) / n)
                + (1.0 - f)
                    * (second_invariant / third_invariant - 3.0
                        + (third_invariant.powf(n) - 1.0) / n)))
    }
}
