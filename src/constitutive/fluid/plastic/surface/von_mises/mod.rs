use super::YieldSurface;
use crate::{
    constitutive::ConstitutiveError,
    math::{ContractWith, Quantity, Tensor, TensorArray},
    mechanics::{FlowDirectionPlastic, MandelStressElastic, StretchingRatePlastic},
    units::{Dissipation, Stress},
};

/// The von Mises yield surface.[^1]
///
/// **Parameters**
/// - None.
///
/// **External variables**
/// - The deviatoric elastic Mandel stress $`\mathbf{M}_\mathrm{e}'`$.
///
/// **Internal variables**
/// - None.
///
/// **Notes**
/// - The equivalent stress is given by
/// ```math
/// \phi(\mathbf{M}_\mathrm{e}') = |\mathbf{M}_\mathrm{e}'|,
/// ```
/// the Frobenius norm of the deviator, and the flow is associative, with the flow direction $`\mathbf{N} = \mathbf{M}_\mathrm{e}'/|\mathbf{M}_\mathrm{e}'|`$.
/// - The equivalent stress is not normalized to the uniaxial stress: in uniaxial tension it is $`\sqrt{2/3}`$ times the axial stress, so the yield stress of a hardening law is $`\sqrt{2/3}`$ times the uniaxial yield stress.
/// - This is the isotropic case of the [Hill surface](super::Hill), with $`F=G=H=1/3`$ and $`L=M=N=1`$.
///
/// [^1]: R. von Mises, Nachr. Ges. Wiss. Göttingen, Math.-Phys. Kl. **4**, 582 (1913).
#[derive(Clone, Debug)]
pub struct VonMises;

impl YieldSurface for VonMises {
    fn equivalent_stress(
        &self,
        deviatoric_mandel_stress: &MandelStressElastic,
    ) -> Result<Quantity<Stress>, ConstitutiveError> {
        Ok(deviatoric_mandel_stress.norm())
    }
    /// ```math
    /// \mathbf{N} = \frac{\mathbf{M}_\mathrm{e}'}{|\mathbf{M}_\mathrm{e}'|}
    /// ```
    fn flow_direction(
        &self,
        deviatoric_mandel_stress: &MandelStressElastic,
    ) -> Result<FlowDirectionPlastic, ConstitutiveError> {
        let magnitude = deviatoric_mandel_stress.norm();
        if magnitude.is_zero() {
            Ok(FlowDirectionPlastic::zero())
        } else {
            Ok(deviatoric_mandel_stress / magnitude)
        }
    }
    /// ```math
    /// \frac{\partial\mathbf{N}}{\partial\mathbf{M}_\mathrm{e}'}:\mathrm{d}\mathbf{M}_\mathrm{e}'
    /// = \frac{1}{|\mathbf{M}_\mathrm{e}'|}\left(\mathrm{d}\mathbf{M}_\mathrm{e}' - \mathbf{N}\,(\mathbf{N}:\mathrm{d}\mathbf{M}_\mathrm{e}')\right)
    /// ```
    fn flow_direction_slope(
        &self,
        deviatoric_mandel_stress: &MandelStressElastic,
        increment: &MandelStressElastic,
    ) -> Result<FlowDirectionPlastic, ConstitutiveError> {
        let magnitude = deviatoric_mandel_stress.norm();
        if magnitude.is_zero() {
            return Ok(FlowDirectionPlastic::zero());
        }
        let direction = deviatoric_mandel_stress / magnitude;
        let slope = increment.contract_with(&direction);
        Ok((increment - direction * slope) / magnitude)
    }
    /// ```math
    /// \phi_\mathrm{d}(\mathbf{D}_\mathrm{p}) = Y\,|\mathbf{D}_\mathrm{p}|
    /// ```
    fn dissipation_potential(
        &self,
        plastic_stretching_rate: StretchingRatePlastic,
        yield_stress: Quantity<Stress>,
    ) -> Result<Quantity<Dissipation>, ConstitutiveError> {
        Ok(yield_stress * plastic_stretching_rate.norm())
    }
}
