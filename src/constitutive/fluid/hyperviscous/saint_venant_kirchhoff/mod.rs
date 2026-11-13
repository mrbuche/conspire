#[cfg(test)]
mod test;

use crate::{
    constitutive::{
        ConstitutiveError,
        fluid::{
            hyperviscous::{Hyperviscous, TWO_THIRDS},
            viscous::Viscous,
        },
    },
    math::{ContractWith, IDENTITY_00, Quantity, Rank2, TensorRank4},
    mechanics::{
        DeformationGradient, DeformationGradientRate, SecondPiolaKirchhoffRateTangentStiffness,
        SecondPiolaKirchhoffStress, SecondPiolaKirchhoffTangentStiffness,
    },
    units::{Dissipation, Viscosity},
};

/// The Saint Venant-Kirchhoff viscous fluid constitutive model.
///
/// **Parameters**
/// - The bulk viscosity $`\zeta`$.
/// - The shear viscosity $`\eta`$.
///
/// **External variables**
/// - The deformation gradient $`\mathbf{F}`$.
/// - The deformation gradient rate $`\dot{\mathbf{F}}`$.
///
/// **Internal variables**
/// - None.
///
/// **Notes**
/// - The Green-Saint Venant strain rate $`\dot{\mathbf{E}}`$ follows from $`\mathbf{E}=\tfrac{1}{2}(\mathbf{C}-\mathbf{1})`$.
#[derive(Clone, Debug)]
pub struct SaintVenantKirchhoff {
    /// The bulk viscosity $`\zeta`$.
    pub bulk_viscosity: Quantity<Viscosity>,
    /// The shear viscosity $`\eta`$.
    pub shear_viscosity: Quantity<Viscosity>,
}

impl Viscous for SaintVenantKirchhoff {
    fn bulk_viscosity(&self) -> Quantity<Viscosity> {
        self.bulk_viscosity
    }
    fn shear_viscosity(&self) -> Quantity<Viscosity> {
        self.shear_viscosity
    }
    /// ```math
    /// \mathbf{S}(\mathbf{F},\dot{\mathbf{F}}) = 2\eta\dot{\mathbf{E}}' + \zeta\,\mathrm{tr}(\dot{\mathbf{E}})\mathbf{1}
    /// ```
    fn viscous_second_piola_kirchhoff_stress(
        &self,
        deformation_gradient: &DeformationGradient,
        deformation_gradient_rate: &DeformationGradientRate,
    ) -> Result<SecondPiolaKirchhoffStress, ConstitutiveError> {
        let strain_rate_term = deformation_gradient_rate.transpose() * deformation_gradient;
        let (deviatoric_strain_rate, strain_rate_trace) =
            ((&strain_rate_term + strain_rate_term.transpose()) * 0.5).deviatoric_and_trace();
        Ok(deviatoric_strain_rate * (2.0 * self.shear_viscosity())
            + IDENTITY_00 * (self.bulk_viscosity() * strain_rate_trace))
    }
    /// ```math
    /// \mathcal{W}_{IJkL}(\mathbf{F}) = \eta\,\delta_{JL}F_{kI} + \eta\,\delta_{IL}F_{kJ} + \left(\zeta - \frac{2}{3}\,\eta\right)\delta_{IJ}F_{kL}
    /// ```
    fn viscous_second_piola_kirchhoff_rate_tangent_stiffness(
        &self,
        deformation_gradient: &DeformationGradient,
        _: &DeformationGradientRate,
    ) -> Result<SecondPiolaKirchhoffRateTangentStiffness, ConstitutiveError> {
        let scaled_deformation_gradient_transpose =
            deformation_gradient.transpose() * self.shear_viscosity();
        Ok(
            TensorRank4::dyad_ik_jl(&scaled_deformation_gradient_transpose, &IDENTITY_00)
                + TensorRank4::dyad_il_jk(&IDENTITY_00, &scaled_deformation_gradient_transpose)
                + TensorRank4::dyad_ij_kl(
                    &(IDENTITY_00 * (self.bulk_viscosity() - TWO_THIRDS * self.shear_viscosity())),
                    deformation_gradient,
                ),
        )
    }
    fn viscous_second_piola_kirchhoff_tangent_stiffness(
        &self,
        _: &DeformationGradient,
        deformation_gradient_rate: &DeformationGradientRate,
    ) -> Result<SecondPiolaKirchhoffTangentStiffness, ConstitutiveError> {
        let scaled_deformation_gradient_rate_transpose =
            deformation_gradient_rate.transpose() * self.shear_viscosity();
        Ok(
            TensorRank4::dyad_ik_jl(&scaled_deformation_gradient_rate_transpose, &IDENTITY_00)
                + TensorRank4::dyad_il_jk(
                    &IDENTITY_00,
                    &scaled_deformation_gradient_rate_transpose,
                )
                + TensorRank4::dyad_ij_kl(
                    &(IDENTITY_00 * (self.bulk_viscosity() - TWO_THIRDS * self.shear_viscosity())),
                    deformation_gradient_rate,
                ),
        )
    }
}

impl Hyperviscous for SaintVenantKirchhoff {
    /// ```math
    /// \psi(\mathbf{F},\dot{\mathbf{F}}) = \eta\,\mathrm{tr}(\dot{\mathbf{E}}^2) + \frac{1}{2}\left(\zeta - \frac{2}{3}\,\eta\right)\mathrm{tr}(\dot{\mathbf{E}})^2
    /// ```
    fn viscous_dissipation(
        &self,
        deformation_gradient: &DeformationGradient,
        deformation_gradient_rate: &DeformationGradientRate,
    ) -> Result<Quantity<Dissipation>, ConstitutiveError> {
        let strain_rate_term = deformation_gradient_rate.transpose() * deformation_gradient;
        let strain_rate = (&strain_rate_term + strain_rate_term.transpose()) * 0.5;
        let strain_rate_trace = strain_rate.trace();
        Ok(
            (&strain_rate * self.shear_viscosity()).contract_with(&strain_rate)
                + (self.bulk_viscosity() - TWO_THIRDS * self.shear_viscosity())
                    * strain_rate_trace
                    * strain_rate_trace
                    * 0.5,
        )
    }
}
