#[cfg(test)]
mod test;

use super::*;
use crate::{
    math::{Quantity, TensorRank4},
    units::{ReciprocalTemperature, Stress, Temperature},
};

/// The Almansi-Hamel thermoelastic solid constitutive model.
///
/// **Parameters**
/// - The bulk modulus $`\kappa`$.
/// - The shear modulus $`\mu`$.
/// - The coefficient of thermal expansion $`\alpha`$.
/// - The reference temperature $`T_\mathrm{ref}`$.
///
/// **External variables**
/// - The deformation gradient $`\mathbf{F}`$.
/// - The temperature $`T`$.
///
/// **Internal variables**
/// - None.
///
/// **Notes**
/// - The Almansi-Hamel strain measure is given by $`\mathbf{e}=\tfrac{1}{2}(\mathbf{1}-\mathbf{B}^{-1})`$.
#[derive(Clone, Debug)]
pub struct AlmansiHamel {
    /// The bulk modulus $`\kappa`$.
    pub bulk_modulus: Quantity<Stress>,
    /// The shear modulus $`\mu`$.
    pub shear_modulus: Quantity<Stress>,
    /// The coefficient of thermal expansion $`\alpha`$.
    pub coefficient_of_thermal_expansion: Quantity<ReciprocalTemperature>,
    /// The reference temperature $`T_\mathrm{ref}`$.
    pub reference_temperature: Quantity<Temperature>,
}

impl Solid for AlmansiHamel {
    fn bulk_modulus(&self) -> Quantity<Stress> {
        self.bulk_modulus
    }
    fn shear_modulus(&self) -> Quantity<Stress> {
        self.shear_modulus
    }
}

impl Thermoelastic for AlmansiHamel {
    /// ```math
    /// \boldsymbol{\sigma}(\mathbf{F}, T) = \frac{2\mu}{J}\,\mathbf{e}' + \frac{\kappa}{J}\,\mathrm{tr}(\mathbf{e})\mathbf{1} - \frac{3\alpha\kappa}{J}(T - T_\mathrm{ref})\mathbf{1}
    /// ```
    fn cauchy_stress(
        &self,
        deformation_gradient: &DeformationGradient,
        temperature: Quantity<Temperature>,
    ) -> Result<CauchyStress, ConstitutiveError> {
        let jacobian = self.jacobian(deformation_gradient)?;
        let inverse_deformation_gradient = deformation_gradient.inverse();
        let strain = (IDENTITY
            - inverse_deformation_gradient.transpose() * &inverse_deformation_gradient)
            * 0.5;
        let (deviatoric_strain, strain_trace) = strain.deviatoric_and_trace();
        Ok(deviatoric_strain * (2.0 * self.shear_modulus() / jacobian)
            + IDENTITY
                * (self.bulk_modulus() / jacobian
                    * (strain_trace
                        - 3.0
                            * self.coefficient_of_thermal_expansion
                            * (temperature - self.reference_temperature()))))
    }
    /// ```math
    /// \begin{aligned}
    /// \mathcal{T}_{ijkL}(\mathbf{F}, T) =
    /// \,& \frac{\mu}{J}\left[B_{jk}^{-1}F_{iL}^{-T} + B_{ik}^{-1}F_{jL}^{-T} - 2e_{ij}'F_{kL}^{-T}\right] + \frac{\lambda}{J}\,\delta_{ij}B_{km}^{-1}F_{mL}^{-T}
    /// \\
    /// &- \frac{\kappa}{J}\Big[\mathrm{tr}(\mathbf{e}) - 3\alpha(T - T_\mathrm{ref})\Big]\delta_{ij}F_{kL}^{-T}
    /// \end{aligned}
    /// ```
    fn cauchy_tangent_stiffness(
        &self,
        deformation_gradient: &DeformationGradient,
        temperature: Quantity<Temperature>,
    ) -> Result<CauchyTangentStiffness, ConstitutiveError> {
        let jacobian = self.jacobian(deformation_gradient)?;
        let inverse_transpose_deformation_gradient = deformation_gradient.inverse_transpose();
        let inverse_left_cauchy_green_deformation = &inverse_transpose_deformation_gradient
            * inverse_transpose_deformation_gradient.transpose();
        let strain = (IDENTITY - &inverse_left_cauchy_green_deformation) * 0.5;
        let (deviatoric_strain, strain_trace) = strain.deviatoric_and_trace();
        Ok((TensorRank4::dyad_il_jk(
            &inverse_transpose_deformation_gradient,
            &inverse_left_cauchy_green_deformation,
        ) + TensorRank4::dyad_ik_jl(
            &inverse_left_cauchy_green_deformation,
            &inverse_transpose_deformation_gradient,
        )) * (self.shear_modulus() / jacobian)
            + TensorRank4::dyad_ij_kl(
                &IDENTITY,
                &(inverse_left_cauchy_green_deformation
                    * &inverse_transpose_deformation_gradient
                    * ((self.bulk_modulus() - self.shear_modulus() * TWO_THIRDS) / jacobian)),
            )
            - TensorRank4::dyad_ij_kl(
                &(deviatoric_strain * (2.0 * self.shear_modulus() / jacobian)
                    + IDENTITY
                        * (self.bulk_modulus() / jacobian
                            * (strain_trace
                                - 3.0
                                    * self.coefficient_of_thermal_expansion
                                    * (temperature - self.reference_temperature())))),
                &inverse_transpose_deformation_gradient,
            ))
    }
    /// ```math
    /// \frac{\partial\boldsymbol{\sigma}}{\partial T}(\mathbf{F}) = -\frac{3\alpha\kappa}{J}\,\mathbf{1}
    /// ```
    fn cauchy_thermal_tangent(
        &self,
        deformation_gradient: &DeformationGradient,
        _: Quantity<Temperature>,
    ) -> Result<CauchyThermalTangent, ConstitutiveError> {
        let jacobian = self.jacobian(deformation_gradient)?;
        Ok(IDENTITY
            * (-3.0 * self.bulk_modulus() * self.coefficient_of_thermal_expansion / jacobian))
    }
    /// ```math
    /// \frac{\partial^2\sigma_{ij}}{\partial F_{kL}\partial T}(\mathbf{F}) = \frac{3\alpha\kappa}{J}\,\delta_{ij}F_{kL}^{-T}
    /// ```
    fn cauchy_mixed_tangent_stiffness(
        &self,
        deformation_gradient: &DeformationGradient,
        _: Quantity<Temperature>,
    ) -> Result<CauchyMixedTangentStiffness, ConstitutiveError> {
        let jacobian = self.jacobian(deformation_gradient)?;
        Ok(
            TensorRank4::dyad_ij_kl(&IDENTITY, &deformation_gradient.inverse_transpose())
                * (3.0 * self.bulk_modulus() * self.coefficient_of_thermal_expansion / jacobian),
        )
    }
    /// ```math
    /// \frac{\partial^2\boldsymbol{\sigma}}{\partial T^2}(\mathbf{F}) = \mathbf{0}
    /// ```
    fn cauchy_thermal_tangent_stiffness(
        &self,
        deformation_gradient: &DeformationGradient,
        _: Quantity<Temperature>,
    ) -> Result<CauchyThermalTangentStiffness, ConstitutiveError> {
        let _jacobian = self.jacobian(deformation_gradient)?;
        Ok(CauchyThermalTangentStiffness::zero())
    }
    fn reference_temperature(&self) -> Quantity<Temperature> {
        self.reference_temperature
    }
}
