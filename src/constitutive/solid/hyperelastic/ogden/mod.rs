#[cfg(test)]
mod test;

use crate::{
    constitutive::{
        ConstitutiveError,
        solid::{Solid, elastic::Elastic, hyperelastic::Hyperelastic},
    },
    math::{
        ContractThirdFourthWithFirstSecond, Current, IDENTITY, Quantity, Rank2, Tensor,
        TensorRank0List, TensorRank2, TensorRank4,
    },
    mechanics::{
        CauchyStress, CauchyTangentStiffness, Deformation, DeformationGradient,
        LeftCauchyGreenDeformation, Scalar,
    },
    units::{EnergyDensity, Stress},
};

/// The Ogden hyperelastic solid constitutive model.[^1]
///
/// **Parameters**
/// - The bulk modulus $`\kappa`$.
/// - The shear moduli $`\mu_n`$ for $`n=1\ldots N`$.
/// - The exponents $`\alpha_n`$ for $`n=1\ldots N`$.
///
/// **External variables**
/// - The deformation gradient $`\mathbf{F}`$.
///
/// **Internal variables**
/// - None.
///
/// **Notes**
/// - The shear modulus is given by $`\mu=\boldsymbol{\alpha}\cdot\boldsymbol{\mu}/2`$.
/// - The Ogden model reduces to the [Neo-Hookean model](super::NeoHookean) when $`N=1`$ and $`\alpha_1\to 2`$.
///
/// [^1]: R.W. Ogden, [Proc. R. Soc. Lond. A **326**, 565 (1972)](https://doi.org/10.1098/rspa.1972.0026).
#[derive(Clone, Debug)]
pub struct Ogden {
    /// The bulk modulus $`\kappa`$.
    pub bulk_modulus: Quantity<Stress>,
    /// The shear moduli $`\mu_n`$.
    pub shear_moduli: Vec<Quantity<Stress>>,
    /// The exponents $`\alpha_n`$.
    pub exponents: Vec<Scalar>,
}

enum Spectrum {
    Eigen(TensorRank0List<3>, LeftCauchyGreenDeformation),
    Fallback(LeftCauchyGreenDeformation),
}

impl Spectrum {
    fn new(tensor: &LeftCauchyGreenDeformation, model: &Ogden) -> Result<Self, ConstitutiveError> {
        if tensor.is_diagonal() || (tensor - &IDENTITY).norm() < 1e-2 {
            Ok(Self::Fallback(tensor.clone()))
        } else {
            let (eigenvalues, eigenvectors) = tensor
                .eigen()
                .map_err(|error| ConstitutiveError::upstream(error, model))?;
            Ok(Self::Eigen(eigenvalues, eigenvectors))
        }
    }
    fn powm(
        &self,
        exponent: Scalar,
        model: &Ogden,
    ) -> Result<LeftCauchyGreenDeformation, ConstitutiveError> {
        match self {
            Self::Eigen(eigenvalues, eigenvectors) => {
                TensorRank2::powm_from_eigen(eigenvalues, eigenvectors, exponent)
            }
            Self::Fallback(tensor) => tensor.powm(exponent),
        }
        .map_err(|error| ConstitutiveError::upstream(error, model))
    }
    fn dpowm(
        &self,
        exponent: Scalar,
        model: &Ogden,
    ) -> Result<TensorRank4<3, Current, Current, Current, Current>, ConstitutiveError> {
        match self {
            Self::Eigen(eigenvalues, eigenvectors) => {
                TensorRank2::dpowm_from_eigen(eigenvalues, eigenvectors, exponent)
            }
            Self::Fallback(tensor) => tensor.dpowm(exponent),
        }
        .map_err(|error| ConstitutiveError::upstream(error, model))
    }
}

impl Solid for Ogden {
    fn bulk_modulus(&self) -> Quantity<Stress> {
        self.bulk_modulus
    }
    fn shear_modulus(&self) -> Quantity<Stress> {
        self.shear_moduli
            .iter()
            .zip(self.exponents.iter())
            .map(|(modulus, exponent)| *modulus * exponent)
            .sum::<Quantity<Stress>>()
            * 0.5
    }
}

impl Elastic for Ogden {
    /// ```math
    /// \boldsymbol{\sigma}(\mathbf{F}) = \sum_{n=1}^N \frac{\mu_n}{J}\,\mathrm{dev}\left({\mathbf{V}^*}^{\alpha_n}\right) + \frac{\kappa}{2}\left(J - \frac{1}{J}\right)\mathbf{1}
    /// ```
    fn cauchy_stress(
        &self,
        deformation_gradient: &DeformationGradient,
    ) -> Result<CauchyStress, ConstitutiveError> {
        let jacobian = self.jacobian(deformation_gradient)?;
        let left_cauchy_green = deformation_gradient.left_cauchy_green();
        let spectrum = Spectrum::new(&left_cauchy_green, self)?;
        let mut cauchy_stress = IDENTITY * self.bulk_modulus() * 0.5 * (jacobian - 1.0 / jacobian);
        for (modulus, exponent) in self.shear_moduli.iter().zip(self.exponents.iter()) {
            let scaling = *modulus / jacobian.powf(exponent / 3.0 + 1.0);
            cauchy_stress += spectrum.powm(exponent / 2.0, self)?.deviatoric() * scaling;
        }
        Ok(cauchy_stress)
    }
    /// ```math
    /// \mathcal{T}_{ijkL}(\mathbf{F}) = \sum_{n=1}^N\frac{\mu_n}{J^{\alpha_n/3+1}}\left[\mathrm{dev}\left(\frac{\partial\mathbf{V}^{\alpha_n}}{\partial F_{kL}}\right)_{ij} - \left(\frac{\alpha_n}{3}+1\right)\mathrm{dev}(\mathbf{V}^{\alpha_n})_{ij}F_{kL}^{-T}\right] + \frac{\kappa}{2}\left(J + \frac{1}{J}\right)\delta_{ij}F_{kL}^{-T}
    /// ```
    fn cauchy_tangent_stiffness(
        &self,
        deformation_gradient: &DeformationGradient,
    ) -> Result<CauchyTangentStiffness, ConstitutiveError> {
        let jacobian = self.jacobian(deformation_gradient)?;
        let left_cauchy_green = deformation_gradient.left_cauchy_green();
        let spectrum = Spectrum::new(&left_cauchy_green, self)?;
        let inverse_transpose_deformation_gradient = deformation_gradient.inverse_transpose();
        let mut cauchy_tangent_stiffness = TensorRank4::dyad_ij_kl(
            &(IDENTITY * (self.bulk_modulus() * 0.5 * (jacobian + 1.0 / jacobian))),
            &inverse_transpose_deformation_gradient,
        );
        for (modulus, exponent) in self.shear_moduli.iter().zip(self.exponents.iter()) {
            let half_exponent = exponent / 2.0;
            let scaling = *modulus / jacobian.powf(exponent / 3.0 + 1.0);
            let scaled_deformation_gradient = deformation_gradient * scaling;
            let raw = spectrum
                .dpowm(half_exponent, self)?
                .contract_third_fourth_with_first_second(
                    &(TensorRank4::dyad_il_jk(&scaled_deformation_gradient, &IDENTITY)
                        + TensorRank4::dyad_ik_jl(&IDENTITY, &scaled_deformation_gradient)),
                );
            let trace_term = spectrum.powm(half_exponent - 1.0, self)?
                * deformation_gradient
                * (scaling * half_exponent * 2.0);
            let cauchy_stress_n = spectrum.powm(half_exponent, self)?.deviatoric() * scaling;
            cauchy_tangent_stiffness += raw
                - TensorRank4::dyad_ij_kl(&(IDENTITY * (1.0 / 3.0)), &trace_term)
                - TensorRank4::dyad_ij_kl(
                    &(cauchy_stress_n * (exponent / 3.0 + 1.0)),
                    &inverse_transpose_deformation_gradient,
                );
        }
        Ok(cauchy_tangent_stiffness)
    }
}

impl Hyperelastic for Ogden {
    /// ```math
    /// a(\mathbf{F}) = \sum_{n=1}^N \frac{\mu_n}{\alpha_n}\big[\mathrm{tr}({\mathbf{V}^*}^{\alpha_n}) - 3\big] + \frac{\kappa}{2}\left[\frac{1}{2}\left(J^2 - 1\right) - \ln J\right]
    /// ```
    fn helmholtz_free_energy_density(
        &self,
        deformation_gradient: &DeformationGradient,
    ) -> Result<Quantity<EnergyDensity>, ConstitutiveError> {
        let jacobian = self.jacobian(deformation_gradient)?;
        let left_cauchy_green = deformation_gradient.left_cauchy_green();
        let spectrum = Spectrum::new(&left_cauchy_green, self)?;
        let mut helmholtz_free_energy_density =
            self.bulk_modulus() * 0.5 * (0.5 * (jacobian.powi(2) - 1.0) - jacobian.ln());
        for (modulus, exponent) in self.shear_moduli.iter().zip(self.exponents.iter()) {
            helmholtz_free_energy_density += (*modulus / *exponent)
                * (spectrum.powm(exponent / 2.0, self)?.trace() / jacobian.powf(exponent / 3.0)
                    - 3.0);
        }
        Ok(helmholtz_free_energy_density)
    }
}
