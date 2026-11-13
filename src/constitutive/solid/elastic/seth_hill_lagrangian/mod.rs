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

/// The Lagrangian Seth-Hill elastic solid constitutive model.[^1]<sup>,</sup>[^2]
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
/// - The Seth-Hill strain measure is given by $`\mathbf{E}^{(m)} = \tfrac{1}{m}(\mathbf{U}^m - \mathbf{1})`$.
/// - The model reduces to the elastic [Hencky model](super::Hencky) when $`m=0`$.
/// - The model reduces to the hyperelastic [Saint Venant-Kirchhoff model](super::super::hyperelastic::SaintVenantKirchhoff) when $`m=2`$.
/// - The model reduces to the [Lagrangian Almansi-Hamel model](super::AlmansiHamelLagrangian) when $`m=-2`$.
/// - This is the Lagrangian counterpart of the [Eulerian Seth-Hill model](super::SethHillEulerian).
///
/// [^1]: B.R. Seth, *Generalized Strain Measure with Applications to Physical Problems*, AD0266913 (1961).
/// [^2]: R. Hill, [J. Mech. Phys. Solids **16**, 229 (1968)](https://doi.org/10.1016/0022-5096(68)90031-8).
#[derive(Clone, Debug)]
pub struct SethHillLagrangian {
    /// The bulk modulus $`\kappa`$.
    pub bulk_modulus: Quantity<Stress>,
    /// The shear modulus $`\mu`$.
    pub shear_modulus: Quantity<Stress>,
    /// The exponent $`m`$.
    pub exponent: Scalar,
}

impl SethHillLagrangian {
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

impl Solid for SethHillLagrangian {
    fn bulk_modulus(&self) -> Quantity<Stress> {
        self.bulk_modulus
    }
    fn shear_modulus(&self) -> Quantity<Stress> {
        self.shear_modulus
    }
}

impl Elastic for SethHillLagrangian {
    /// ```math
    /// \mathbf{S}(\mathbf{F}) = 2\mu\mathbf{E}^{(m)}{}' + \kappa\,\mathrm{tr}\big(\mathbf{E}^{(m)}\big)\mathbf{1}
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
        let (deviatoric_strain, strain_trace) = ((right_cauchy_green
            .powm(0.5 * self.exponent())
            .map_err(|error| ConstitutiveError::upstream(error, self))?
            - IDENTITY_00)
            / self.exponent())
        .deviatoric_and_trace();
        Ok(deviatoric_strain * (2.0 * self.shear_modulus())
            + IDENTITY_00 * (self.bulk_modulus() * strain_trace))
    }
    /// ```math
    /// \mathcal{G}_{IJkL}(\mathbf{F}) = \mu\,\frac{\partial C_{IJ}^{m/2}}{\partial F_{kL}} + \lambda\,\delta_{IJ}F_{kM}C_{ML}^{m/2-1}
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
            &deformation_gradient_transpose * (2.0 * self.shear_modulus() / self.exponent());
        let trace_term = deformation_gradient
            * spectrum
                .powm(half_exponent - 1.0)
                .map_err(|error| ConstitutiveError::upstream(error, self))?;
        Ok(spectrum
            .dpowm(half_exponent)
            .map_err(|error| ConstitutiveError::upstream(error, self))?
            .contract_third_fourth_with_first_second(
                &(TensorRank4::dyad_il_jk(&IDENTITY_00, &scaled_deformation_gradient_transpose)
                    + TensorRank4::dyad_ik_jl(
                        &scaled_deformation_gradient_transpose,
                        &IDENTITY_00,
                    )),
            )
            + TensorRank4::dyad_ij_kl(
                &(IDENTITY_00 * (self.bulk_modulus() - TWO_THIRDS * self.shear_modulus())),
                &trace_term,
            ))
    }
}
