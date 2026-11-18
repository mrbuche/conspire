#[cfg(test)]
mod test;

use crate::math::{
    Derivative, Differentiable, Quantity, Scalar, Tensor, TensorArray, TensorVec,
    integrate::{FixedStep, Implicit, IntegrationError, OdeIntegrator},
};
use core::{
    fmt::Debug,
    ops::{Mul, Sub},
};

/// Implicit, two-stage, second-order, fixed-step, Runge-Kutta method.[^1]
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
/// k_2 = f(t_{n+1}, y_{n+1})
/// ```
/// ```math
/// y_{n+1} = y_n + \frac{h}{2}\left(k_1 + k_2\right)
/// ```
///
/// [^1]: Also known as the implicit trapezoidal rule.
#[derive(Debug, Default)]
pub struct Trapezoidal {
    /// Fixed value for the time step.
    dt: Scalar,
}

impl<Y, U> OdeIntegrator<Y, U> for Trapezoidal
where
    Y: Tensor,
    U: TensorVec<Item = Y>,
{
}

impl<T> FixedStep<T> for Trapezoidal {
    fn dt(&self) -> Quantity<T> {
        Quantity::new(self.dt)
    }
}

impl<Y, J, U, V, T> Implicit<Y, J, U, V, T> for Trapezoidal
where
    Y: Differentiable<T> + Tensor,
    Derivative<Y, T>: Mul<Quantity<T>, Output = Y>,
    J: Differentiable<T> + Tensor + TensorArray,
    Derivative<J, T>: Mul<Quantity<T>, Output = J>,
    for<'a> &'a Y: Sub<&'a Y, Output = Y>,
    U: TensorVec<Item = Y>,
    V: TensorVec<Item = Derivative<Y, T>>,
{
    fn residual(
        &self,
        mut function: impl FnMut(Quantity<T>, &Y) -> Result<Derivative<Y, T>, IntegrationError>,
        t: Quantity<T>,
        y: &Y,
        t_trial: Quantity<T>,
        y_trial: &Y,
        dt: Quantity<T>,
    ) -> Result<Y, String> {
        Ok(y_trial - y - (function(t, y)? + function(t_trial, y_trial)?) * (0.5 * dt))
    }
    fn hessian(
        &self,
        mut jacobian: impl FnMut(Quantity<T>, &Y) -> Result<Derivative<J, T>, IntegrationError>,
        _t: Quantity<T>,
        _y: &Y,
        t_trial: Quantity<T>,
        y_trial: &Y,
        dt: Quantity<T>,
    ) -> Result<J, String> {
        Ok(J::identity() - jacobian(t_trial, y_trial)? * (0.5 * dt))
    }
}
