#[cfg(test)]
mod test;

use crate::{
    math::{
        Derivative, Differentiable, Quantity, Scalar, Tensor, TensorVec,
        integrate::{IntegrationError, OdeIntegrator, StabilityRegion, Times},
    },
    units::Time,
};
use core::f64::consts::FRAC_PI_2;

pub(crate) mod fixed_step;
pub(crate) mod variable_step;

/// The fastest time scale of a right-hand side function, which bounds a stable time step.
///
/// The time scale is the reciprocal of the spectral radius of the Jacobian.
#[derive(Clone, Copy, Debug)]
pub enum Spectrum<T = Time> {
    /// The eigenvalues are real and negative, as in diffusion.
    Real(Quantity<T>),
    /// The eigenvalues are imaginary, as in waves.
    Imaginary(Quantity<T>),
    /// The eigenvalues lie along a ray at the given angle from the negative real axis,
    /// as in damped waves, between $`0`$ (diffusion) and $`\pi/2`$ (waves).
    Complex(Quantity<T>, Scalar),
    /// The eigenvalues are real and negative at the first time scale and imaginary at the
    /// second, as in diffusion coupled to waves.
    Mixed(Quantity<T>, Quantity<T>),
}

impl<T> Spectrum<T> {
    pub(crate) fn limit(self, stability: &mut Stability) -> Quantity<T> {
        match self {
            Self::Real(scale) => scale * stability.extent(0.0),
            Self::Imaginary(scale) => scale * stability.extent(FRAC_PI_2),
            Self::Complex(scale, angle) => scale * stability.extent(angle),
            Self::Mixed(real, imaginary) => {
                let real = real * stability.extent(0.0);
                let imaginary = imaginary * stability.extent(FRAC_PI_2);
                if real < imaginary { real } else { imaginary }
            }
        }
    }
}

pub(crate) struct Stability {
    region: StabilityRegion,
    angle: Scalar,
    extent: Scalar,
}

impl Stability {
    pub(crate) fn new(region: StabilityRegion) -> Self {
        Self {
            region,
            angle: Scalar::NAN,
            extent: 0.0,
        }
    }
    fn extent(&mut self, angle: Scalar) -> Scalar {
        if angle != self.angle {
            self.angle = angle;
            self.extent = self.region.extent(angle);
        }
        self.extent
    }
}

pub(crate) fn check_safety(safety: Scalar) -> Result<(), IntegrationError> {
    if safety > 0.0 && safety <= 1.0 {
        Ok(())
    } else {
        Err(IntegrationError::InvalidSafetyFactor(safety))
    }
}

/// Explicit integrators for ordinary differential equations.
pub trait Explicit<Y, U, V, T = Time>
where
    Self: OdeIntegrator<Y, U>,
    Y: Differentiable<T> + Tensor,
    U: TensorVec<Item = Y>,
    V: TensorVec<Item = Derivative<Y, T>>,
{
    const SLOPES: usize;
    /// Solves an initial value problem by explicitly integrating a system of ordinary differential equations.
    ///
    /// ```math
    /// \frac{dy}{dt} = f(t, y),\quad y(t_0) = y_0
    /// ```
    fn integrate(
        &self,
        function: impl FnMut(Quantity<T>, &Y) -> Result<Derivative<Y, T>, String>,
        time: &[Quantity<T>],
        initial_condition: Y,
    ) -> Result<(Times<T>, U, V), IntegrationError>;
    /// Solves an initial value problem like [`integrate`](Self::integrate), subject to a stability bound.
    ///
    /// At each step, `bound` reports the fastest time scale of the right-hand side at the current state, and where its eigenvalues lie in the complex plane. The stability limit is that time scale times the extent of the method's stability region along the reported direction. The time step may use at most the fraction `safety` of that limit, where `safety` is in $`(0, 1]`$. A fixed time step above it is an error, and an adaptive time step is reduced to it.
    fn integrate_bounded(
        &self,
        function: impl FnMut(Quantity<T>, &Y) -> Result<Derivative<Y, T>, String>,
        bound: impl FnMut(Quantity<T>, &Y) -> Result<Spectrum<T>, String>,
        safety: Scalar,
        time: &[Quantity<T>],
        initial_condition: Y,
    ) -> Result<(Times<T>, U, V), IntegrationError>;
}
