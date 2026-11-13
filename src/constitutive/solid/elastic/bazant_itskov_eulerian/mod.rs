#[cfg(test)]
mod test;

use crate::{
    constitutive::{
        ConstitutiveError,
        solid::{Solid, TWO_THIRDS, elastic::Elastic},
    },
    math::{ContractThirdFourthWithFirstSecond, IDENTITY, Quantity, Rank2, Spectrum, TensorRank4},
    mechanics::{CauchyStress, CauchyTangentStiffness, Deformation, DeformationGradient, Scalar},
    units::Stress,
};

/// The Eulerian Bažant-Itskov elastic solid constitutive model.[^1]<sup>,</sup>[^2]
///
/// **Parameters**
/// - The bulk modulus $`\kappa`$.
/// - The shear modulus $`\mu`$.
/// - The exponent $`m`$.
///
/// **External variables**
/// - The deformation gradient $`\mathbf{F}`$.
///
/// **Internal variables**
/// - None.
///
/// **Notes**
/// - The Eulerian Bažant-Itskov strain measure is given by $`\mathbf{h}^{(m)} = \tfrac{1}{2m}(\mathbf{V}^m - \mathbf{V}^{-m})`$.
/// - The model reduces to the hyperelastic [Hencky model](super::super::hyperelastic::Hencky) when $`m=0`$.
/// - This is the Eulerian counterpart of the [Lagrangian Bažant-Itskov model](super::BazantItskovLagrangian).
///
/// [^1]: Z.P. Bažant, [J. Eng. Mater. Technol. **120**, 131 (1998)](https://doi.org/10.1115/1.2807001).
/// [^2]: M. Itskov, [Mech. Res. Commun. **31**, 507 (2004)](https://doi.org/10.1016/j.mechrescom.2004.02.006).
#[derive(Clone, Debug)]
pub struct BazantItskovEulerian {
    /// The bulk modulus $`\kappa`$.
    pub bulk_modulus: Quantity<Stress>,
    /// The shear modulus $`\mu`$.
    pub shear_modulus: Quantity<Stress>,
    /// The exponent $`m`$.
    pub exponent: Scalar,
}

impl BazantItskovEulerian {
    /// Returns the exponent.
    pub fn exponent(&self) -> Scalar {
        self.exponent
    }
}

impl Solid for BazantItskovEulerian {
    fn bulk_modulus(&self) -> Quantity<Stress> {
        self.bulk_modulus
    }
    fn shear_modulus(&self) -> Quantity<Stress> {
        self.shear_modulus
    }
}

impl Elastic for BazantItskovEulerian {
    /// ```math
    /// \boldsymbol{\sigma}(\mathbf{F}) = \frac{2\mu}{J}\,\mathbf{h}^{(m)}{}' + \frac{\kappa}{J}\,\mathrm{tr}\big(\mathbf{h}^{(m)}\big)\mathbf{1}
    /// ```
    fn cauchy_stress(
        &self,
        deformation_gradient: &DeformationGradient,
    ) -> Result<CauchyStress, ConstitutiveError> {
        let jacobian = self.jacobian(deformation_gradient)?;
        let left_cauchy_green = deformation_gradient.left_cauchy_green();
        let (deviatoric_strain, strain_trace) = if self.exponent() == 0.0 {
            (left_cauchy_green
                .logm()
                .map_err(|error| ConstitutiveError::upstream(error, self))?
                * 0.5)
                .deviatoric_and_trace()
        } else {
            let half_exponent = 0.5 * self.exponent();
            let spectrum = Spectrum::new(&left_cauchy_green)
                .map_err(|error| ConstitutiveError::upstream(error, self))?;
            ((spectrum
                .powm(half_exponent)
                .map_err(|error| ConstitutiveError::upstream(error, self))?
                - spectrum
                    .powm(-half_exponent)
                    .map_err(|error| ConstitutiveError::upstream(error, self))?)
                / (2.0 * self.exponent()))
            .deviatoric_and_trace()
        };
        Ok(deviatoric_strain * (2.0 * self.shear_modulus() / jacobian)
            + IDENTITY * (self.bulk_modulus() * strain_trace / jacobian))
    }
    /// ```math
    /// \mathcal{T}_{ijkL}(\mathbf{F}) = \frac{\mu}{J}\,\frac{\partial\big(B_{ij}^{m/2} - B_{ij}^{-m/2}\big)}{\partial F_{kL}} + \frac{\lambda}{2J}\,\delta_{ij}\big(B_{km}^{m/2-1} + B_{km}^{-m/2-1}\big)F_{mL} - \sigma_{ij}F_{kL}^{-T}
    /// ```
    fn cauchy_tangent_stiffness(
        &self,
        deformation_gradient: &DeformationGradient,
    ) -> Result<CauchyTangentStiffness, ConstitutiveError> {
        let jacobian = self.jacobian(deformation_gradient)?;
        let left_cauchy_green = deformation_gradient.left_cauchy_green();
        let inverse_transpose_deformation_gradient = deformation_gradient.inverse_transpose();
        if self.exponent() == 0.0 {
            let (deviatoric_strain, strain_trace) = (left_cauchy_green
                .logm()
                .map_err(|error| ConstitutiveError::upstream(error, self))?
                * 0.5)
                .deviatoric_and_trace();
            let scaled_deformation_gradient =
                deformation_gradient * (self.shear_modulus() / jacobian);
            return Ok((left_cauchy_green
                .dlogm()
                .map_err(|error| ConstitutiveError::upstream(error, self))?
                .contract_third_fourth_with_first_second(
                    &(TensorRank4::dyad_il_jk(&scaled_deformation_gradient, &IDENTITY)
                        + TensorRank4::dyad_ik_jl(&IDENTITY, &scaled_deformation_gradient)),
                ))
                + TensorRank4::dyad_ij_kl(
                    &(IDENTITY
                        * ((self.bulk_modulus() - TWO_THIRDS * self.shear_modulus()) / jacobian)
                        - deviatoric_strain * (2.0 * self.shear_modulus() / jacobian)
                        - IDENTITY * (self.bulk_modulus() * strain_trace / jacobian)),
                    &inverse_transpose_deformation_gradient,
                ));
        }
        let half_exponent = 0.5 * self.exponent();
        let spectrum = Spectrum::new(&left_cauchy_green)
            .map_err(|error| ConstitutiveError::upstream(error, self))?;
        let (deviatoric_strain, strain_trace) = ((spectrum
            .powm(half_exponent)
            .map_err(|error| ConstitutiveError::upstream(error, self))?
            - spectrum
                .powm(-half_exponent)
                .map_err(|error| ConstitutiveError::upstream(error, self))?)
            / (2.0 * self.exponent()))
        .deviatoric_and_trace();
        let cauchy_stress = deviatoric_strain * (2.0 * self.shear_modulus() / jacobian)
            + IDENTITY * (self.bulk_modulus() * strain_trace / jacobian);
        let scaled_deformation_gradient =
            deformation_gradient * (self.shear_modulus() / (self.exponent() * jacobian));
        let trace_term = (spectrum
            .powm(half_exponent - 1.0)
            .map_err(|error| ConstitutiveError::upstream(error, self))?
            + spectrum
                .powm(-half_exponent - 1.0)
                .map_err(|error| ConstitutiveError::upstream(error, self))?)
            * deformation_gradient
            * 0.5;
        Ok((spectrum
            .dpowm(half_exponent)
            .map_err(|error| ConstitutiveError::upstream(error, self))?
            - spectrum
                .dpowm(-half_exponent)
                .map_err(|error| ConstitutiveError::upstream(error, self))?)
        .contract_third_fourth_with_first_second(
            &(TensorRank4::dyad_il_jk(&scaled_deformation_gradient, &IDENTITY)
                + TensorRank4::dyad_ik_jl(&IDENTITY, &scaled_deformation_gradient)),
        ) + TensorRank4::dyad_ij_kl(
            &(IDENTITY * ((self.bulk_modulus() - TWO_THIRDS * self.shear_modulus()) / jacobian)),
            &trace_term,
        ) - TensorRank4::dyad_ij_kl(&cauchy_stress, &inverse_transpose_deformation_gradient))
    }
}
