#[cfg(test)]
mod test;

use crate::{
    constitutive::{
        ConstitutiveError, FromParameters,
        solid::{
            Solid, TWO_THIRDS,
            elastic::{Elastic, FRAC_1_SQRT_3, SQRT_2_3, StiffnessBound},
            hyperelastic::Hyperelastic,
        },
    },
    math::{IDENTITY_00, Quantity, Rank2, Tensor, TensorRank4},
    mechanics::{
        Deformation, DeformationGradient, SecondPiolaKirchhoffStress,
        SecondPiolaKirchhoffTangentStiffness,
    },
    units::{EnergyDensity, Stress},
};

/// The Saint Venant-Kirchhoff hyperelastic solid constitutive model.
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
/// - The Green-Saint Venant strain measure is given by $`\mathbf{E}=\tfrac{1}{2}(\mathbf{C}-\mathbf{1})`$.
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

impl FromParameters for SaintVenantKirchhoff {
    const NAME: &'static str = "saint_venant_kirchhoff";
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

impl Elastic for SaintVenantKirchhoff {
    /// ```math
    /// \mathbf{S}(\mathbf{F}) = 2\mu\mathbf{E}' + \kappa\,\mathrm{tr}(\mathbf{E})\mathbf{1}
    /// ```
    fn second_piola_kirchhoff_stress(
        &self,
        deformation_gradient: &DeformationGradient,
    ) -> Result<SecondPiolaKirchhoffStress, ConstitutiveError> {
        let _jacobian = self.jacobian(deformation_gradient)?;
        let (deviatoric_strain, strain_trace) =
            ((deformation_gradient.right_cauchy_green() - IDENTITY_00) * 0.5)
                .deviatoric_and_trace();
        Ok(deviatoric_strain * (2.0 * self.shear_modulus())
            + IDENTITY_00 * (self.bulk_modulus() * strain_trace))
    }
    /// ```math
    /// \mathcal{G}_{IJkL}(\mathbf{F}) = \mu\,\delta_{JL}F_{kI} + \mu\,\delta_{IL}F_{kJ} + \lambda\,\delta_{IJ}F_{kL}
    /// ```
    fn second_piola_kirchhoff_tangent_stiffness(
        &self,
        deformation_gradient: &DeformationGradient,
    ) -> Result<SecondPiolaKirchhoffTangentStiffness, ConstitutiveError> {
        let _jacobian = self.jacobian(deformation_gradient)?;
        let scaled_deformation_gradient_transpose =
            deformation_gradient.transpose() * self.shear_modulus();
        Ok(
            TensorRank4::dyad_ik_jl(&scaled_deformation_gradient_transpose, &IDENTITY_00)
                + TensorRank4::dyad_il_jk(&IDENTITY_00, &scaled_deformation_gradient_transpose)
                + TensorRank4::dyad_ij_kl(
                    &(IDENTITY_00 * (self.bulk_modulus() - TWO_THIRDS * self.shear_modulus())),
                    deformation_gradient,
                ),
        )
    }
}

impl StiffnessBound for SaintVenantKirchhoff {
    /// ```math
    /// \hat\lambda = \frac{\lambda^+}{J}\,a\left(a + \frac{b}{\sqrt{3}}\right)
    /// ```
    /// ```math
    /// \hat\mu = \frac{1}{2J}\left[2\mu\left(a + \sqrt{\tfrac{2}{3}}\,b\right)^2 + \lambda^+ b\left(b + \sqrt{3}\,a\right)\right] + \frac{s}{2}
    /// ```
    /// where $`a=\mathrm{tr}(\mathbf{B})/3`$, $`b=|\mathbf{B}'|`$, $`\lambda^+=\max(0,\lambda)`$,
    /// and $`s=\max[0,\,\mathrm{tr}(\boldsymbol{\sigma})/3 + \sqrt{2/3}\,|\boldsymbol{\sigma}'|]`$.
    fn stiffness_bound(
        &self,
        deformation_gradient: &DeformationGradient,
    ) -> Result<(Quantity<Stress>, Quantity<Stress>), ConstitutiveError> {
        let jacobian = self.jacobian(deformation_gradient)?;
        let (deviator, trace) = deformation_gradient
            .left_cauchy_green()
            .deviatoric_and_trace();
        let (mean, spread) = (trace.value() / 3.0, deviator.norm().value());
        let largest = mean + SQRT_2_3 * spread;
        let (stress_deviator, stress_trace) = self
            .cauchy_stress(deformation_gradient)?
            .deviatoric_and_trace();
        let tension =
            (stress_trace / 3.0 + stress_deviator.norm() * SQRT_2_3).max(Quantity::default());
        let lame =
            (self.bulk_modulus() - self.shear_modulus() * TWO_THIRDS).max(Quantity::default());
        Ok((
            lame * (mean * (mean + spread * FRAC_1_SQRT_3) / jacobian),
            (self.shear_modulus() * (largest * largest)
                + lame * (0.5 * spread * (spread + mean / FRAC_1_SQRT_3)))
                / jacobian
                + tension * 0.5,
        ))
    }
}

impl Hyperelastic for SaintVenantKirchhoff {
    /// ```math
    /// a(\mathbf{F}) = \mu\,\mathrm{tr}(\mathbf{E}^2) + \frac{\lambda}{2}\,\mathrm{tr}(\mathbf{E})^2
    /// ```
    fn helmholtz_free_energy_density(
        &self,
        deformation_gradient: &DeformationGradient,
    ) -> Result<Quantity<EnergyDensity>, ConstitutiveError> {
        let _jacobian = self.jacobian(deformation_gradient)?;
        let strain = (deformation_gradient.right_cauchy_green() - IDENTITY_00) * 0.5;
        Ok(self.shear_modulus() * strain.squared_trace()
            + 0.5
                * (self.bulk_modulus() - TWO_THIRDS * self.shear_modulus())
                * strain.trace().powi(2))
    }
}
