#[cfg(test)]
mod test;

use crate::math::{
    Derivative, Differentiable, Quantity, Scalar, Tensor, TensorVec,
    integrate::{
        Explicit, FixedStep, FixedStepExplicit, IntegrationError, OdeIntegrator, Spectrum, Times,
        ode::explicit::variable_step::bogacki_shampine::Tableau as BogackiShampineTableau,
    },
};
use core::ops::{Div, Mul, Sub};

/// Explicit, three-stage, third-order, fixed-step, Runge-Kutta method.[^1]
///
/// ```math
/// \frac{dy}{dt} = f(t, y)
/// ```
/// ```math
/// t_{n+1} = t_n + h
/// ```
/// ```math
/// k_1 = f(t_n, y_n)
/// ```
/// ```math
/// k_2 = f(t_n + \tfrac{1}{2} h, y_n + \tfrac{1}{2} h k_1)
/// ```
/// ```math
/// k_3 = f(t_n + \tfrac{3}{4} h, y_n + \tfrac{3}{4} h k_2)
/// ```
/// ```math
/// y_{n+1} = y_n + \frac{h}{9}\left(2k_1 + 3k_2 + 4k_3\right)
/// ```
///
/// [^1]: Fixed-step variant of the [Bogacki-Shampine](`crate::math::integrate::BogackiShampine`) method.
#[derive(Debug, Default)]
pub struct BogackiShampine {
    /// Fixed value for the time step.
    dt: Scalar,
}

impl<Y, U> OdeIntegrator<Y, U> for BogackiShampine
where
    Y: Tensor,
    U: TensorVec<Item = Y>,
{
}

impl<T> FixedStep<T> for BogackiShampine {
    fn dt(&self) -> Quantity<T> {
        Quantity::new(self.dt)
    }
}

impl<Y, U, V, T> Explicit<Y, U, V, T> for BogackiShampine
where
    Y: Differentiable<T> + Div<Quantity<T>, Output = Derivative<Y, T>> + Tensor,
    Derivative<Y, T>: Mul<Quantity<T>, Output = Y>,
    for<'a> &'a Y: Mul<Scalar, Output = Y> + Sub<&'a Y, Output = Y>,
    for<'a> &'a Derivative<Y, T>:
        Mul<Scalar, Output = Derivative<Y, T>> + Mul<Quantity<T>, Output = Y>,
    U: TensorVec<Item = Y>,
    V: TensorVec<Item = Derivative<Y, T>>,
{
    const SLOPES: usize = 3;
    fn integrate(
        &self,
        function: impl FnMut(Quantity<T>, &Y) -> Result<Derivative<Y, T>, String>,
        time: &[Quantity<T>],
        initial_condition: Y,
    ) -> Result<(Times<T>, U, V), IntegrationError> {
        self.integrate_fixed_step(function, time, initial_condition)
    }
    fn integrate_bounded(
        &self,
        function: impl FnMut(Quantity<T>, &Y) -> Result<Derivative<Y, T>, String>,
        bound: impl FnMut(Quantity<T>, &Y) -> Result<Spectrum<T>, String>,
        safety: Scalar,
        time: &[Quantity<T>],
        initial_condition: Y,
    ) -> Result<(Times<T>, U, V), IntegrationError> {
        self.integrate_fixed_step_bounded(function, bound, safety, time, initial_condition)
    }
}

impl<Y, U, V, T> FixedStepExplicit<Y, U, V, T> for BogackiShampine
where
    Y: Differentiable<T> + Div<Quantity<T>, Output = Derivative<Y, T>> + Tensor,
    Derivative<Y, T>: Mul<Quantity<T>, Output = Y>,
    for<'a> &'a Y: Mul<Scalar, Output = Y> + Sub<&'a Y, Output = Y>,
    for<'a> &'a Derivative<Y, T>:
        Mul<Scalar, Output = Derivative<Y, T>> + Mul<Quantity<T>, Output = Y>,
    U: TensorVec<Item = Y>,
    V: TensorVec<Item = Derivative<Y, T>>,
{
    type Tableau = BogackiShampineTableau;
}
