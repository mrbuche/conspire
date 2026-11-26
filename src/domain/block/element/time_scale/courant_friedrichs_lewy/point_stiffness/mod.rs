use crate::{
    constitutive::{
        ConstitutiveError,
        solid::elastic::{SQRT_2_3, StiffnessBound},
    },
    math::{Quantity, Rank2, Reference, Tensor, TensorRank2},
    mechanics::DeformationGradient,
    units::{LengthPerMass, RateSquared},
};

/// The weight of an integration point times the sum over the nodes of the outer product of the
/// reference gradient vector of each with itself, over its lumped mass.
pub type GradientGram = TensorRank2<3, Reference, Reference, LengthPerMass>;

/// How the stiffness at an integration point bounds the angular frequencies of an element.
pub trait PointStiffness<C> {
    /// A bound on the squared angular frequency that an integration point contributes.
    fn squared_frequency(
        &self,
        constitutive_model: &C,
        deformation_gradient: &DeformationGradient,
        gradient_gram: &GradientGram,
    ) -> Result<Quantity<RateSquared>, ConstitutiveError>;
}

/// The stiffness as the moduli of the [bound of the model](StiffnessBound) have it, which needs
/// no tangent stiffness or eigenvalue.
///
/// With $`\mathbf{G}`$ the gradient Gram tensor pushed to the current configuration,
/// ```math
/// J\left[\hat\lambda\,\mathrm{tr}(\mathbf{G}) + 2\hat\mu\left(\mathrm{tr}(\mathbf{G})/3 + \sqrt{2/3}\,|\mathbf{G}'|\right)\right]
/// ```
/// where the term in parentheses is not less than the largest eigenvalue of $`\mathbf{G}`$.
#[derive(Clone, Copy, Debug, Default)]
pub struct ModelBound;

impl<C> PointStiffness<C> for ModelBound
where
    C: StiffnessBound,
{
    fn squared_frequency(
        &self,
        constitutive_model: &C,
        deformation_gradient: &DeformationGradient,
        gradient_gram: &GradientGram,
    ) -> Result<Quantity<RateSquared>, ConstitutiveError> {
        let (first, second) = constitutive_model.stiffness_bound(deformation_gradient)?;
        let inverse_transpose = deformation_gradient.inverse_transpose();
        let (deviator, trace) =
            (&inverse_transpose * gradient_gram * inverse_transpose.transpose())
                .deviatoric_and_trace();
        let largest = trace / 3.0 + deviator.norm() * SQRT_2_3;
        Ok((first * trace + second * largest * 2.0) * deformation_gradient.determinant())
    }
}
