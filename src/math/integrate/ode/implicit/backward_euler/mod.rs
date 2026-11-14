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

/// Implicit, single-stage, first-order, fixed-step, Runge-Kutta method.[^1]
///
/// ```math
/// \frac{dy}{dt} = f(t, y)
/// ```
/// ```math
/// t_{n+1} = t_n + h
/// ```
/// ```math
/// k_1 = f(t_{n+1}, y_{n+1})
/// ```
/// ```math
/// y_{n+1} = y_n + hk_1
/// ```
///
/// [^1]: Also known as the backward Euler method.
#[derive(Debug, Default)]
pub struct BackwardEuler {
    /// Fixed value for the time step.
    pub dt: Scalar,
}

impl<Y, U> OdeIntegrator<Y, U> for BackwardEuler
where
    Y: Tensor,
    U: TensorVec<Item = Y>,
{
}

impl<T> FixedStep<T> for BackwardEuler {
    fn dt(&self) -> Quantity<T> {
        Quantity::new(self.dt)
    }
}

impl<Y, J, U, V, T> Implicit<Y, J, U, V, T> for BackwardEuler
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
        _t: Quantity<T>,
        y: &Y,
        t_trial: Quantity<T>,
        y_trial: &Y,
        dt: Quantity<T>,
    ) -> Result<Y, String> {
        Ok(y_trial - y - function(t_trial, y_trial)? * dt)
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
        Ok(J::identity() - jacobian(t_trial, y_trial)? * dt)
    }
}
