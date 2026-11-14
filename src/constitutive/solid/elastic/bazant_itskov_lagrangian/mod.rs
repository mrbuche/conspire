#[cfg(test)]
mod test;

use super::hencky::Hencky;
use crate::{
    constitutive::{
        ConstitutiveError,
        solid::{Solid, TWO_THIRDS, elastic::Elastic},
    },
    math::{
        ContractThirdFourthWithFirstSecond, IDENTITY_00, Quantity, Rank2, Spectrum, TensorRank4,
    },
    mechanics::{
        Deformation, DeformationGradient, Scalar, SecondPiolaKirchhoffStress,
        SecondPiolaKirchhoffTangentStiffness,
    },
    units::Stress,
};

/// The Lagrangian Bažant-Itskov elastic solid constitutive model.[^1]<sup>,</sup>[^2]
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
/// - The Bažant-Itskov strain measure is given by $`\mathbf{h}^{(m)} = \tfrac{1}{2m}(\mathbf{U}^m - \mathbf{U}^{-m})`$.
/// - The model reduces to the elastic [Hencky model](super::Hencky) when $`m=0`$.
/// - This is the Lagrangian counterpart of the [Eulerian Bažant-Itskov model](super::BazantItskovEulerian).
///
/// [^1]: Z.P. Bažant, [J. Eng. Mater. Technol. **120**, 131 (1998)](https://doi.org/10.1115/1.2807001).
/// [^2]: M. Itskov, [Mech. Res. Commun. **31**, 507 (2004)](https://doi.org/10.1016/j.mechrescom.2004.02.006).
#[derive(Clone, Debug)]
pub struct BazantItskovLagrangian {
    /// The bulk modulus $`\kappa`$.
    pub bulk_modulus: Quantity<Stress>,
    /// The shear modulus $`\mu`$.
    pub shear_modulus: Quantity<Stress>,
    /// The exponent $`m`$.
    pub exponent: Scalar,
}

impl BazantItskovLagrangian {
    /// Returns the exponent.
    pub fn exponent(&self) -> Scalar {
        self.exponent
    }
    fn hencky(&self) -> Hencky {
        Hencky {
            bulk_modulus: self.bulk_modulus,
            shear_modulus: self.shear_modulus,
        }
    }
}

impl Solid for BazantItskovLagrangian {
    fn bulk_modulus(&self) -> Quantity<Stress> {
        self.bulk_modulus
    }
    fn shear_modulus(&self) -> Quantity<Stress> {
        self.shear_modulus
    }
}

impl Elastic for BazantItskovLagrangian {
    /// ```math
    /// \mathbf{S}(\mathbf{F}) = 2\mu\mathbf{h}^{(m)}{}' + \kappa\,\mathrm{tr}\big(\mathbf{h}^{(m)}\big)\mathbf{1}
    /// ```
    fn second_piola_kirchhoff_stress(
        &self,
        deformation_gradient: &DeformationGradient,
    ) -> Result<SecondPiolaKirchhoffStress, ConstitutiveError> {
        if self.exponent() == 0.0 {
            return self
                .hencky()
                .second_piola_kirchhoff_stress(deformation_gradient);
        }
        let _jacobian = self.jacobian(deformation_gradient)?;
        let right_cauchy_green = deformation_gradient.right_cauchy_green();
        let half_exponent = 0.5 * self.exponent();
        let spectrum = Spectrum::new(&right_cauchy_green)
            .map_err(|error| ConstitutiveError::upstream(error, self))?;
        let (deviatoric_strain, strain_trace) = ((spectrum
            .powm(half_exponent)
            .map_err(|error| ConstitutiveError::upstream(error, self))?
            - spectrum
                .powm(-half_exponent)
                .map_err(|error| ConstitutiveError::upstream(error, self))?)
            / (2.0 * self.exponent()))
        .deviatoric_and_trace();
        Ok(deviatoric_strain * (2.0 * self.shear_modulus())
            + IDENTITY_00 * (self.bulk_modulus() * strain_trace))
    }
    /// ```math
    /// \mathcal{G}_{IJkL}(\mathbf{F}) = \mu\,\frac{\partial\big(C_{IJ}^{m/2} - C_{IJ}^{-m/2}\big)}{\partial F_{kL}} + \frac{\lambda}{2}\,\delta_{IJ}F_{kM}\big(C_{ML}^{m/2-1} + C_{ML}^{-m/2-1}\big)
    /// ```
    fn second_piola_kirchhoff_tangent_stiffness(
        &self,
        deformation_gradient: &DeformationGradient,
    ) -> Result<SecondPiolaKirchhoffTangentStiffness, ConstitutiveError> {
        if self.exponent() == 0.0 {
            return self
                .hencky()
                .second_piola_kirchhoff_tangent_stiffness(deformation_gradient);
        }
        let _jacobian = self.jacobian(deformation_gradient)?;
        let right_cauchy_green = deformation_gradient.right_cauchy_green();
        let half_exponent = 0.5 * self.exponent();
        let spectrum = Spectrum::new(&right_cauchy_green)
            .map_err(|error| ConstitutiveError::upstream(error, self))?;
        let deformation_gradient_transpose = deformation_gradient.transpose();
        let scaled_deformation_gradient_transpose =
            &deformation_gradient_transpose * (self.shear_modulus() / self.exponent());
        let trace_term = deformation_gradient
            * (spectrum
                .powm(half_exponent - 1.0)
                .map_err(|error| ConstitutiveError::upstream(error, self))?
                + spectrum
                    .powm(-half_exponent - 1.0)
                    .map_err(|error| ConstitutiveError::upstream(error, self))?)
            * 0.5;
        Ok((spectrum
            .dpowm(half_exponent)
            .map_err(|error| ConstitutiveError::upstream(error, self))?
            - spectrum
                .dpowm(-half_exponent)
                .map_err(|error| ConstitutiveError::upstream(error, self))?)
        .contract_third_fourth_with_first_second(
            &(TensorRank4::dyad_il_jk(&IDENTITY_00, &scaled_deformation_gradient_transpose)
                + TensorRank4::dyad_ik_jl(&scaled_deformation_gradient_transpose, &IDENTITY_00)),
        ) + TensorRank4::dyad_ij_kl(
            &(IDENTITY_00 * (self.bulk_modulus() - TWO_THIRDS * self.shear_modulus())),
            &trace_term,
        ))
    }
}
