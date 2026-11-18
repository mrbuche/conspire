use super::YieldSurface;
use crate::{
    constitutive::ConstitutiveError,
    math::{ContractWith, Intermediate, Quantity, Tensor, TensorArray, TensorRank1, TensorRank2},
    mechanics::{FlowDirectionPlastic, MandelStressElastic, Scalar, StretchingRatePlastic},
    units::{Dimensionless, Dissipation, Rate, Stress},
};

/// The Hill yield surface for orthotropic plasticity.[^1]
///
/// **Parameters**
/// - The coefficient $`F`$.
/// - The coefficient $`G`$.
/// - The coefficient $`H`$.
/// - The coefficient $`L`$.
/// - The coefficient $`M`$.
/// - The coefficient $`N`$.
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
/// \phi^2 = F(a_{22}-a_{33})^2 + G(a_{33}-a_{11})^2 + H(a_{11}-a_{22})^2 + 2L\,a_{23}^2 + 2M\,a_{31}^2 + 2N\,a_{12}^2,
/// \qquad \mathbf{a} = \mathrm{sym}(\mathbf{M}_\mathrm{e}'),
/// ```
/// in the axes of the intermediate configuration, and the coefficients must be positive.
/// - The surface is the finite-strain extension, in the deviatoric Mandel stress, of the quadratic criterion of Hill, and the flow is associative.
/// - The coefficients are normalized so that $`F=G=H=1/3`$ and $`L=M=N=1`$ gives $`\phi=|\mathbf{M}_\mathrm{e}'|`$, reducing the surface to the [von Mises surface](super::VonMises). The yield stress is given by the hardening law it is combined with.
///
/// [^1]: R. Hill, [Proc. R. Soc. Lond. A **193**, 281 (1948)](https://doi.org/10.1098/rspa.1948.0045).
#[derive(Clone, Debug)]
pub struct Hill {
    /// The coefficient $`F`$.
    pub f: Scalar,
    /// The coefficient $`G`$.
    pub g: Scalar,
    /// The coefficient $`H`$.
    pub h: Scalar,
    /// The coefficient $`L`$.
    pub l: Scalar,
    /// The coefficient $`M`$.
    pub m: Scalar,
    /// The coefficient $`N`$.
    pub n: Scalar,
}

impl Hill {
    /// The quadratic form as a symmetric operator, $`\mathcal{H}:\mathbf{a}`$.
    fn operator<I, U>(&self, a: &TensorRank2<3, I, I, U>) -> TensorRank2<3, I, I, U> {
        let Self { f, g, h, l, m, n } = *self;
        let (a_11, a_22, a_33) = (a[0][0], a[1][1], a[2][2]);
        TensorRank2::from([
            [
                (a_11 - a_33) * g + (a_11 - a_22) * h,
                a[0][1] * n,
                a[0][2] * m,
            ],
            [
                a[0][1] * n,
                (a_22 - a_33) * f + (a_22 - a_11) * h,
                a[1][2] * l,
            ],
            [
                a[0][2] * m,
                a[1][2] * l,
                (a_33 - a_22) * f + (a_33 - a_11) * g,
            ],
        ])
    }
    fn equivalent(&self, a: &MandelStressElastic) -> Quantity<Stress> {
        a.contract_with(&self.operator(a))
            .max(Quantity::default())
            .sqrt()
    }
}

impl YieldSurface for Hill {
    fn equivalent_stress(
        &self,
        deviatoric_mandel_stress: &MandelStressElastic,
    ) -> Result<Quantity<Stress>, ConstitutiveError> {
        Ok(self.equivalent(&deviatoric_mandel_stress.symmetric_part()))
    }
    /// ```math
    /// \mathbf{N} = \frac{\mathcal{H}:\mathbf{a}}{\phi}
    /// ```
    fn flow_direction(
        &self,
        deviatoric_mandel_stress: &MandelStressElastic,
    ) -> Result<FlowDirectionPlastic, ConstitutiveError> {
        let a = deviatoric_mandel_stress.symmetric_part();
        let magnitude = self.equivalent(&a);
        if magnitude.is_zero() {
            Ok(FlowDirectionPlastic::zero())
        } else {
            Ok(self.operator(&a) / magnitude)
        }
    }
    /// ```math
    /// \frac{\partial\mathbf{N}}{\partial\mathbf{M}_\mathrm{e}'}:\mathrm{d}\mathbf{a}
    /// = \frac{1}{\phi}\left(\mathcal{H}:\mathrm{d}\mathbf{a} - \mathbf{N}\,(\mathbf{N}:\mathrm{d}\mathbf{a})\right)
    /// ```
    fn flow_direction_slope(
        &self,
        deviatoric_mandel_stress: &MandelStressElastic,
        increment: &MandelStressElastic,
    ) -> Result<FlowDirectionPlastic, ConstitutiveError> {
        let a = deviatoric_mandel_stress.symmetric_part();
        let magnitude = self.equivalent(&a);
        if magnitude.is_zero() {
            return Ok(FlowDirectionPlastic::zero());
        }
        let direction = self.operator(&a) / magnitude;
        let d_a = increment.symmetric_part();
        let slope = d_a.contract_with(&direction);
        Ok((self.operator(&d_a) - &(direction * slope)) / magnitude)
    }
    /// ```math
    /// \phi_\mathrm{d}(\mathbf{D}_\mathrm{p}) = Y\sqrt{\mathbf{D}_\mathrm{p}:\mathcal{H}^+:\mathbf{D}_\mathrm{p}}
    /// ```
    fn dissipation_potential(
        &self,
        plastic_stretching_rate: StretchingRatePlastic,
        yield_stress: Quantity<Stress>,
    ) -> Result<Quantity<Dissipation>, ConstitutiveError> {
        let Self { f, g, h, l, m, n } = *self;
        let d = plastic_stretching_rate.symmetric_part();
        let normal = TensorRank2::<3, Intermediate, Intermediate, Dimensionless>::from([
            [g + h + 1.0, 1.0 - h, 1.0 - g],
            [1.0 - h, f + h + 1.0, 1.0 - f],
            [1.0 - g, 1.0 - f, f + g + 1.0],
        ]);
        let rates = TensorRank1::<3, Intermediate, Rate>::from([d[0][0], d[1][1], d[2][2]]);
        let solved = normal.inverse() * &rates;
        let dual = TensorRank2::<3, Intermediate, Intermediate, Rate>::from([
            [solved[0], d[0][1] / n, d[0][2] / m],
            [d[0][1] / n, solved[1], d[1][2] / l],
            [d[0][2] / m, d[1][2] / l, solved[2]],
        ]);
        let squared = d.contract_with(&dual);
        Ok(yield_stress * squared.max(Quantity::default()).sqrt())
    }
}
