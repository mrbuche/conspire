#[cfg(test)]
mod test;

use super::*;
use crate::{
    constitutive::solid::{FIVE_THIRDS, TWO_THIRDS},
    math::{IDENTITY, Quantity, Rank2, TensorRank4},
    mechanics::{
        CauchyMixedTangentStiffness, CauchyThermalTangent, CauchyThermalTangentStiffness,
        Deformation,
    },
    units::{
        Dimensionless, EnergyDensity, ReciprocalTemperature, ReciprocalTemperatureSquared, Stress,
        Temperature,
    },
};

/// The Neo-Hookean thermohyperelastic solid constitutive model.[^1]
///
/// **Parameters**
/// - The bulk modulus $`\kappa`$.
/// - The shear modulus $`\mu`$ at the reference temperature.
/// - The coefficient of thermal expansion $`\alpha`$ at the reference temperature.
/// - The change of the coefficient of thermal expansion with temperature $`\alpha'`$.
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
/// - The shear modulus is proportional to the temperature, $`\mu(T)=\mu\,T/T_\mathrm{ref}`$, so that the shear response is purely entropic, as in an ideal rubber: its internal energy does not depend on the deformation, stretching it at constant length raises its stress with temperature, and stretching it adiabatically warms it. The thermal expansion enters only through the volumetric response, with the coefficient $`\alpha(T) = \alpha + \alpha'(T - T_\mathrm{ref})`$ and the thermal strain $`\int_{T_\mathrm{ref}}^{T}\alpha(\theta)\,\mathrm{d}\theta`$, which is $`\alpha(T - T_\mathrm{ref})`$ when $`\alpha' = 0`$.
///
/// [^1]: R.S. Rivlin, [Philos. Trans. R. Soc. London, Ser. A **240**, 459 (1948)](https://doi.org/10.1098/rsta.1948.0002).
#[derive(Clone, Debug)]
pub struct NeoHookean {
    /// The bulk modulus $`\kappa`$.
    pub bulk_modulus: Quantity<Stress>,
    /// The shear modulus $`\mu`$ at the reference temperature.
    pub shear_modulus: Quantity<Stress>,
    /// The coefficient of thermal expansion $`\alpha`$ at the reference temperature.
    pub coefficient_of_thermal_expansion: Quantity<ReciprocalTemperature>,
    /// The change of the coefficient of thermal expansion with temperature $`\alpha'`$.
    pub coefficient_of_thermal_expansion_slope: Quantity<ReciprocalTemperatureSquared>,
    /// The reference temperature $`T_\mathrm{ref}`$.
    pub reference_temperature: Quantity<Temperature>,
}

impl Solid for NeoHookean {
    fn bulk_modulus(&self) -> Quantity<Stress> {
        self.bulk_modulus
    }
    fn shear_modulus(&self) -> Quantity<Stress> {
        self.shear_modulus
    }
}

impl NeoHookean {
    fn shear_modulus_at(&self, temperature: Quantity<Temperature>) -> Quantity<Stress> {
        self.shear_modulus() * (temperature / self.reference_temperature())
    }
    fn thermal_expansion_at(
        &self,
        temperature: Quantity<Temperature>,
    ) -> Quantity<ReciprocalTemperature> {
        self.coefficient_of_thermal_expansion
            + self.coefficient_of_thermal_expansion_slope
                * (temperature - self.reference_temperature())
    }
    fn thermal_strain_at(&self, temperature: Quantity<Temperature>) -> Quantity<Dimensionless> {
        (self.coefficient_of_thermal_expansion
            + self.coefficient_of_thermal_expansion_slope
                * (temperature - self.reference_temperature())
                * 0.5)
            * (temperature - self.reference_temperature())
    }
}

impl Thermoelastic for NeoHookean {
    /// ```math
    /// \boldsymbol{\sigma}(\mathbf{F}, T) = \frac{\mu\,T}{T_\mathrm{ref}\,J^{5/3}}\,\mathbf{B}' + \kappa\left[\frac{1}{2}\left(J - \frac{1}{J}\right) - 3\left(\alpha + \frac{\alpha'}{2}(T - T_\mathrm{ref})\right)(T - T_\mathrm{ref})\right]\mathbf{1}
    /// ```
    fn cauchy_stress(
        &self,
        deformation_gradient: &DeformationGradient,
        temperature: Quantity<Temperature>,
    ) -> Result<CauchyStress, ConstitutiveError> {
        let jacobian = self.jacobian(deformation_gradient)?;
        Ok(
            deformation_gradient.left_cauchy_green().deviatoric() / jacobian.powf(FIVE_THIRDS)
                * self.shear_modulus_at(temperature)
                + IDENTITY
                    * (self.bulk_modulus()
                        * (0.5 * (jacobian - 1.0 / jacobian)
                            - 3.0 * self.thermal_strain_at(temperature))),
        )
    }
    /// ```math
    /// \mathcal{T}_{ijkL}(\mathbf{F}, T) = \frac{\mu\,T}{T_\mathrm{ref}\,J^{5/3}}\left(\delta_{ik}F_{jL} + \delta_{jk}F_{iL} - \frac{2}{3}\delta_{ij}F_{kL}\right) + \left[\frac{\kappa}{2}\left(J + \frac{1}{J}\right)\delta_{ij} - \frac{5}{3}\frac{\mu\,T}{T_\mathrm{ref}\,J^{5/3}}B_{ij}'\right]F_{kL}^{-T}
    /// ```
    fn cauchy_tangent_stiffness(
        &self,
        deformation_gradient: &DeformationGradient,
        temperature: Quantity<Temperature>,
    ) -> Result<CauchyTangentStiffness, ConstitutiveError> {
        let jacobian = self.jacobian(deformation_gradient)?;
        let inverse_transpose_deformation_gradient = deformation_gradient.inverse_transpose();
        let scaled_shear_modulus = self.shear_modulus_at(temperature) / jacobian.powf(FIVE_THIRDS);
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
    /// ```math
    /// \frac{\partial\boldsymbol{\sigma}}{\partial T}(\mathbf{F}, T) = \frac{\mu}{T_\mathrm{ref}\,J^{5/3}}\,\mathbf{B}' - 3\kappa\left[\alpha + \alpha'(T - T_\mathrm{ref})\right]\mathbf{1}
    /// ```
    fn cauchy_thermal_tangent(
        &self,
        deformation_gradient: &DeformationGradient,
        temperature: Quantity<Temperature>,
    ) -> Result<CauchyThermalTangent, ConstitutiveError> {
        let jacobian = self.jacobian(deformation_gradient)?;
        Ok(
            deformation_gradient.left_cauchy_green().deviatoric() / jacobian.powf(FIVE_THIRDS)
                * (self.shear_modulus() / self.reference_temperature())
                - IDENTITY * (3.0 * self.bulk_modulus() * self.thermal_expansion_at(temperature)),
        )
    }
    /// ```math
    /// \frac{\partial^2\sigma_{ij}}{\partial F_{kL}\partial T}(\mathbf{F}) = \frac{\mu}{T_\mathrm{ref}\,J^{5/3}}\left(\delta_{ik}F_{jL} + F_{iL}\delta_{jk} - \frac{2}{3}\,\delta_{ij}F_{kL} - \frac{5}{3}\,B'_{ij}F_{kL}^{-T}\right)
    /// ```
    fn cauchy_mixed_tangent_stiffness(
        &self,
        deformation_gradient: &DeformationGradient,
        _: Quantity<Temperature>,
    ) -> Result<CauchyMixedTangentStiffness, ConstitutiveError> {
        let jacobian = self.jacobian(deformation_gradient)?;
        let scaled_shear_modulus_rate =
            self.shear_modulus() / self.reference_temperature() / jacobian.powf(FIVE_THIRDS);
        Ok((TensorRank4::dyad_ik_jl(&IDENTITY, deformation_gradient)
            + TensorRank4::dyad_il_jk(deformation_gradient, &IDENTITY)
            - TensorRank4::dyad_ij_kl(&IDENTITY, deformation_gradient) * (TWO_THIRDS))
            * scaled_shear_modulus_rate
            - TensorRank4::dyad_ij_kl(
                &(deformation_gradient.left_cauchy_green().deviatoric()
                    * (scaled_shear_modulus_rate * FIVE_THIRDS)),
                &deformation_gradient.inverse_transpose(),
            ))
    }
    /// ```math
    /// \frac{\partial^2\boldsymbol{\sigma}}{\partial T^2}(\mathbf{F}) = -3\kappa\alpha'\,\mathbf{1}
    /// ```
    fn cauchy_thermal_tangent_stiffness(
        &self,
        deformation_gradient: &DeformationGradient,
        _: Quantity<Temperature>,
    ) -> Result<CauchyThermalTangentStiffness, ConstitutiveError> {
        let _jacobian = self.jacobian(deformation_gradient)?;
        Ok(IDENTITY * (-3.0 * self.bulk_modulus() * self.coefficient_of_thermal_expansion_slope))
    }
    fn reference_temperature(&self) -> Quantity<Temperature> {
        self.reference_temperature
    }
}

impl Thermohyperelastic for NeoHookean {
    /// ```math
    /// a(\mathbf{F}, T) = \frac{\mu\,T}{2\,T_\mathrm{ref}}\left[\mathrm{tr}(\mathbf{B}^*) - 3\right] + \frac{\kappa}{2}\left[\frac{1}{2}\left(J^2 - 1\right) - \ln J\right] - 3\kappa\left(\alpha + \frac{\alpha'}{2}(T - T_\mathrm{ref})\right)(T - T_\mathrm{ref})(J - 1)
    /// ```
    fn helmholtz_free_energy_density(
        &self,
        deformation_gradient: &DeformationGradient,
        temperature: Quantity<Temperature>,
    ) -> Result<Quantity<EnergyDensity>, ConstitutiveError> {
        let jacobian = self.jacobian(deformation_gradient)?;
        Ok(0.5
            * (self.shear_modulus_at(temperature)
                * (deformation_gradient.left_cauchy_green().trace() / jacobian.powf(TWO_THIRDS)
                    - 3.0)
                + self.bulk_modulus() * (0.5 * (jacobian.powi(2) - 1.0) - jacobian.ln()))
            - self.bulk_modulus() * (3.0 * self.thermal_strain_at(temperature)) * (jacobian - 1.0))
    }
}
