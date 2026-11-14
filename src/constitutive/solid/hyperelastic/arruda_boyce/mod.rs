#[cfg(test)]
mod test;
use crate::math::Quantity;
use crate::math::TensorRank4;

use crate::{
    constitutive::{
        ConstitutiveError,
        solid::{FIVE_THIRDS, Solid, TWO_THIRDS, elastic::Elastic, hyperelastic::Hyperelastic},
    },
    math::{
        IDENTITY, Rank2,
        special::{inverse_langevin, langevin_derivative},
    },
    mechanics::{CauchyStress, CauchyTangentStiffness, Deformation, DeformationGradient, Scalar},
    units::{EnergyDensity, Stress},
};

/// The Arruda-Boyce hyperelastic solid constitutive model.[^1]
///
/// **Parameters**
/// - The bulk modulus $`\kappa`$.
/// - The shear modulus $`\mu`$.
/// - The number of links $`N_b`$.
///
/// **External variables**
/// - The deformation gradient $`\mathbf{F}`$.
///
/// **Internal variables**
/// - None.
///
/// **Notes**
/// - The nondimensional end-to-end length per link of a chain is $`\gamma=\sqrt{\mathrm{tr}(\mathbf{B}^*)/3N_b}`$.
/// - The nondimensional force is given by the inverse Langevin function as $`\eta=\mathcal{L}^{-1}(\gamma)`$.
/// - The initial values are given by $`\gamma_0=\sqrt{1/3N_b}`$ and $`\eta_0=\mathcal{L}^{-1}(\gamma_0)`$.
/// - The Arruda-Boyce model reduces to the [Neo-Hookean model](super::NeoHookean) when $`N_b\to\infty`$.
///
/// [^1]: E.M. Arruda and M.C. Boyce, [J. Mech. Phys. Solids **41**, 389 (1993)](https://doi.org/10.1016/0022-5096(93)90013-6).
#[derive(Clone, Debug)]
pub struct ArrudaBoyce {
    /// The bulk modulus $`\kappa`$.
    pub bulk_modulus: Quantity<Stress>,
    /// The shear modulus $`\mu`$.
    pub shear_modulus: Quantity<Stress>,
    /// The number of links $`N_b`$.
    pub number_of_links: Scalar,
}

impl ArrudaBoyce {
    /// Returns the number of links.
    pub fn number_of_links(&self) -> Scalar {
        self.number_of_links
    }
}

impl Solid for ArrudaBoyce {
    fn bulk_modulus(&self) -> Quantity<Stress> {
        self.bulk_modulus
    }
    fn shear_modulus(&self) -> Quantity<Stress> {
        self.shear_modulus
    }
}

impl Elastic for ArrudaBoyce {
    /// ```math
    /// \boldsymbol{\sigma}(\mathbf{F}) = \frac{\mu\gamma_0\eta}{J\gamma\eta_0}\,{\mathbf{B}^*}' + \frac{\kappa}{2}\left(J - \frac{1}{J}\right)\mathbf{1}
    /// ```
    fn cauchy_stress(
        &self,
        deformation_gradient: &DeformationGradient,
    ) -> Result<CauchyStress, ConstitutiveError> {
        let jacobian = self.jacobian(deformation_gradient)?;
        let (
            deviatoric_isochoric_left_cauchy_green_deformation,
            isochoric_left_cauchy_green_deformation_trace,
        ) = (deformation_gradient.left_cauchy_green() / jacobian.powf(TWO_THIRDS))
            .deviatoric_and_trace();
        let gamma = (isochoric_left_cauchy_green_deformation_trace / 3.0 / self.number_of_links())
            .sqrt()
            .value();
        if gamma >= 1.0 {
            Err(ConstitutiveError::custom(
                "Maximum extensibility reached.",
                self,
            ))
        } else {
            let gamma_0 = (1.0 / self.number_of_links()).sqrt();
            Ok(deviatoric_isochoric_left_cauchy_green_deformation
                * (self.shear_modulus() * inverse_langevin(gamma) / inverse_langevin(gamma_0)
                    * gamma_0
                    / gamma
                    / jacobian)
                + IDENTITY * self.bulk_modulus() * 0.5 * (jacobian - 1.0 / jacobian))
        }
    }
    /// ```math
    /// \begin{aligned}
    /// \mathcal{T}_{ijkL}(\mathbf{F}) =
    /// \,& \frac{\mu\gamma_0\eta}{J^{5/3}\gamma\eta_0}\left(\delta_{ik}F_{jL} + \delta_{jk}F_{iL} - \frac{2}{3}\,\delta_{ij}F_{kL}- \frac{5}{3} \, B_{ij}'F_{kL}^{-T} \right)
    /// \\
    /// &+ \frac{\mu\gamma_0\eta}{3J^{7/3}N_b\gamma^2\eta_0}\left(\frac{1}{\eta\mathcal{L}'(\eta)} - \frac{1}{\gamma}\right)B_{ij}'B_{km}'F_{mL}^{-T} + \frac{\kappa}{2} \left(J + \frac{1}{J}\right)\delta_{ij}F_{kL}^{-T}
    /// \end{aligned}
    /// ```
    fn cauchy_tangent_stiffness(
        &self,
        deformation_gradient: &DeformationGradient,
    ) -> Result<CauchyTangentStiffness, ConstitutiveError> {
        let jacobian = self.jacobian(deformation_gradient)?;
        let inverse_transpose_deformation_gradient = deformation_gradient.inverse_transpose();
        let left_cauchy_green_deformation = deformation_gradient.left_cauchy_green();
        let deviatoric_left_cauchy_green_deformation = left_cauchy_green_deformation.deviatoric();
        let (
            deviatoric_isochoric_left_cauchy_green_deformation,
            isochoric_left_cauchy_green_deformation_trace,
        ) = (left_cauchy_green_deformation / jacobian.powf(TWO_THIRDS)).deviatoric_and_trace();
        let gamma = (isochoric_left_cauchy_green_deformation_trace / 3.0 / self.number_of_links())
            .sqrt()
            .value();
        if gamma >= 1.0 {
            Err(ConstitutiveError::custom(
                "Maximum extensibility reached.",
                self,
            ))
        } else {
            let gamma_0 = (1.0 / self.number_of_links()).sqrt();
            let eta = inverse_langevin(gamma);
            let scaled_shear_modulus =
                gamma_0 / inverse_langevin(gamma_0) * self.shear_modulus() * eta
                    / gamma
                    / jacobian.powf(FIVE_THIRDS);
            let scaled_deviatoric_isochoric_left_cauchy_green_deformation =
                deviatoric_left_cauchy_green_deformation * scaled_shear_modulus;
            let term = TensorRank4::dyad_ij_kl(
                &scaled_deviatoric_isochoric_left_cauchy_green_deformation,
                &(deviatoric_isochoric_left_cauchy_green_deformation
                    * &inverse_transpose_deformation_gradient
                    * ((1.0 / eta / langevin_derivative(eta) - 1.0 / gamma)
                        / 3.0
                        / self.number_of_links()
                        / gamma)),
            );
            Ok((TensorRank4::dyad_ik_jl(&IDENTITY, deformation_gradient)
                + TensorRank4::dyad_il_jk(deformation_gradient, &IDENTITY)
                - TensorRank4::dyad_ij_kl(&IDENTITY, deformation_gradient) * (TWO_THIRDS))
                * scaled_shear_modulus
                + TensorRank4::dyad_ij_kl(
                    &(IDENTITY * (0.5 * self.bulk_modulus() * (jacobian + 1.0 / jacobian))
                        - scaled_deviatoric_isochoric_left_cauchy_green_deformation
                            * (FIVE_THIRDS)),
                    &inverse_transpose_deformation_gradient,
                )
                + term)
        }
    }
}

impl Hyperelastic for ArrudaBoyce {
    /// ```math
    /// a(\mathbf{F}) = \frac{3\mu N_b\gamma_0}{\eta_0}\left[\gamma\eta - \gamma_0\eta_0 - \ln\left(\frac{\eta_0\sinh\eta}{\eta\sinh\eta_0}\right) \right] + \frac{\kappa}{2}\left[\frac{1}{2}\left(J^2 - 1\right) - \ln J\right]
    /// ```
    fn helmholtz_free_energy_density(
        &self,
        deformation_gradient: &DeformationGradient,
    ) -> Result<Quantity<EnergyDensity>, ConstitutiveError> {
        let jacobian = self.jacobian(deformation_gradient)?;
        let isochoric_left_cauchy_green_deformation =
            deformation_gradient.left_cauchy_green() / jacobian.powf(TWO_THIRDS);
        let gamma =
            (isochoric_left_cauchy_green_deformation.trace() / 3.0 / self.number_of_links())
                .sqrt()
                .value();
        if gamma >= 1.0 {
            Err(ConstitutiveError::custom(
                "Maximum extensibility reached.",
                self,
            ))
        } else {
            let eta = inverse_langevin(gamma);
            let gamma_0 = (1.0 / self.number_of_links()).sqrt();
            let eta_0 = inverse_langevin(gamma_0);
            Ok(3.0 * gamma_0 / eta_0
                * self.shear_modulus()
                * self.number_of_links()
                * (gamma * eta
                    - gamma_0 * eta_0
                    - (eta_0 * eta.sinh() / (eta * eta_0.sinh())).ln())
                + 0.5 * self.bulk_modulus() * (0.5 * (jacobian.powi(2) - 1.0) - jacobian.ln()))
        }
    }
}
