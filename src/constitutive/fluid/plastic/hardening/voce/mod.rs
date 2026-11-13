use super::PlasticHardening;
use crate::{constitutive::ConstitutiveError, math::Quantity, mechanics::Scalar, units::Stress};

/// The Voce isotropic hardening law.[^1]
///
/// **Parameters**
/// - The initial yield stress $`Y_0`$.
/// - The linear hardening slope $`H`$.
/// - The saturation stress $`Q`$.
/// - The saturation rate $`b`$.
///
/// **External variables**
/// - The equivalent plastic strain $`\varepsilon_\mathrm{p}`$.
///
/// **Internal variables**
/// - None.
///
/// **Notes**
/// - The yield stress is given by
/// ```math
/// Y(\varepsilon_\mathrm{p}) = Y_0 + H\,\varepsilon_\mathrm{p} + Q\left(1 - e^{-b\,\varepsilon_\mathrm{p}}\right),
/// ```
/// and the hardening modulus by $`\mathrm{d}Y/\mathrm{d}\varepsilon_\mathrm{p} = H + Qb\,e^{-b\,\varepsilon_\mathrm{p}}`$.
/// - The linear term vanishes for $`H=0`$, which is the Voce law proper, and the yield stress then saturates at $`Y_0 + Q`$.
/// - The hardening modulus starts at $`H+Qb`$ and approaches $`H`$ as the plastic strain grows.
/// - The law is independent of the yield surface it is combined with.
///
/// [^1]: E. Voce, J. Inst. Met. **74**, 537 (1948).
#[derive(Clone, Debug)]
pub struct Voce {
    /// The initial yield stress $`Y_0`$.
    pub yield_stress: Quantity<Stress>,
    /// The linear hardening slope $`H`$, which persists after saturation.
    pub hardening_slope: Quantity<Stress>,
    /// The saturation stress $`Q`$ added to the yield stress at full saturation.
    pub saturation_stress: Quantity<Stress>,
    /// The saturation rate $`b`$.
    pub saturation_rate: Scalar,
}

impl PlasticHardening for Voce {
    fn initial_yield_stress(&self) -> Quantity<Stress> {
        self.yield_stress
    }
    fn yield_stress(
        &self,
        equivalent_plastic_strain: Quantity,
    ) -> Result<Quantity<Stress>, ConstitutiveError> {
        let decay = (equivalent_plastic_strain * -self.saturation_rate).exp();
        Ok(
            self.yield_stress + self.saturation_stress - self.saturation_stress * decay
                + self.hardening_slope * equivalent_plastic_strain,
        )
    }
    fn hardening_modulus(
        &self,
        equivalent_plastic_strain: Quantity,
    ) -> Result<Quantity<Stress>, ConstitutiveError> {
        let decay = (equivalent_plastic_strain * -self.saturation_rate).exp();
        Ok(self.hardening_slope + self.saturation_stress * decay * self.saturation_rate)
    }
}
