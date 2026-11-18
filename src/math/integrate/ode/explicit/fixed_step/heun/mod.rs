#[cfg(test)]
mod test;

use crate::math::{
    Derivative, Differentiable, Quantity, Scalar, Tensor, TensorVec,
    integrate::{
        ButcherTableau, Explicit, FixedStep, FixedStepExplicit, IntegrationError, OdeIntegrator,
        Spectrum, Times,
    },
};
use core::ops::{Add, Mul};

/// The Heun (explicit trapezoidal) tableau.
#[derive(Debug)]
pub struct Tableau;

impl ButcherTableau for Tableau {
    const STAGES: usize = 2;
    const ORDER: Scalar = 2.0;
    const A: &'static [&'static [Scalar]] = &[&[], &[1.0]];
    const C: &'static [Scalar] = &[0.0, 1.0];
    const B: &'static [Scalar] = &[0.5, 0.5];
}

/// Explicit, two-stage, second-order, fixed-step, Runge-Kutta method.[^1]
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
/// k_2 = f(t_{n+1}, y_n + h k_1)
/// ```
/// ```math
/// y_{n+1} = y_n + \frac{h}{2}\left(k_1 + k_2\right)
/// ```
///
/// [^1]: Also known as the Heun method or the explicit trapezoidal rule.
#[derive(Debug, Default)]
pub struct Heun {
    /// Fixed value for the time step.
    dt: Scalar,
}

impl<Y, U> OdeIntegrator<Y, U> for Heun
where
    Y: Tensor,
    U: TensorVec<Item = Y>,
{
}

impl<T> FixedStep<T> for Heun {
    fn dt(&self) -> Quantity<T> {
        Quantity::new(self.dt)
    }
}

impl<Y, U, V, T> Explicit<Y, U, V, T> for Heun
where
    Y: Differentiable<T> + Tensor,
    Derivative<Y, T>: Mul<Quantity<T>, Output = Y>,
    for<'a> &'a Derivative<Y, T>:
        Add<&'a Derivative<Y, T>, Output = Derivative<Y, T>> + Mul<Quantity<T>, Output = Y>,
    U: TensorVec<Item = Y>,
    V: TensorVec<Item = Derivative<Y, T>>,
{
    const SLOPES: usize = 2;
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

impl<Y, U, V, T> FixedStepExplicit<Y, U, V, T> for Heun
where
    Y: Differentiable<T> + Tensor,
    Derivative<Y, T>: Mul<Quantity<T>, Output = Y>,
    for<'a> &'a Derivative<Y, T>:
        Add<&'a Derivative<Y, T>, Output = Derivative<Y, T>> + Mul<Quantity<T>, Output = Y>,
    U: TensorVec<Item = Y>,
    V: TensorVec<Item = Derivative<Y, T>>,
{
    type Tableau = Tableau;
}
