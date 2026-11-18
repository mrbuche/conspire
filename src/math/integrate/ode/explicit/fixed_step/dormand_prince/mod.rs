#[cfg(test)]
mod test;

use crate::math::{
    Derivative, Differentiable, Quantity, Scalar, Tensor, TensorVec,
    integrate::{
        Explicit, FixedStep, FixedStepExplicit, IntegrationError, OdeIntegrator, Spectrum, Times,
        ode::explicit::variable_step::dormand_prince::Tableau as DormandPrinceTableau,
    },
};
use core::ops::{Mul, Sub};

/// Explicit, six-stage, fifth-order, fixed-step, Runge-Kutta method.[^1]
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
/// k_2 = f(t_n + \tfrac{1}{5} h, y_n + \tfrac{1}{5} h k_1)
/// ```
/// ```math
/// k_3 = f(t_n + \tfrac{3}{10} h, y_n + \tfrac{3}{40} h k_1 + \tfrac{9}{40} h k_2)
/// ```
/// ```math
/// k_4 = f(t_n + \tfrac{4}{5} h, y_n + \tfrac{44}{45} h k_1 - \tfrac{56}{15} h k_2 + \tfrac{32}{9} h k_3)
/// ```
/// ```math
/// k_5 = f(t_n + \tfrac{8}{9} h, y_n + \tfrac{19372}{6561} h k_1 - \tfrac{25360}{2187} h k_2 + \tfrac{64448}{6561} h k_3 - \tfrac{212}{729} h k_4)
/// ```
/// ```math
/// k_6 = f(t_n + h, y_n + \tfrac{9017}{3168} h k_1 - \tfrac{355}{33} h k_2 - \tfrac{46732}{5247} h k_3 + \tfrac{49}{176} h k_4 - \tfrac{5103}{18656} h k_5)
/// ```
/// ```math
/// y_{n+1} = y_n + h\left(\frac{35}{384}\,k_1 + \frac{500}{1113}\,k_3 + \frac{125}{192}\,k_4 - \frac{2187}{6784}\,k_5 + \frac{11}{84}\,k_6\right)
/// ```
///
/// [^1]: Fixed-step variant of the [Dormand-Prince](`crate::math::integrate::DormandPrince`) method.
#[derive(Debug, Default)]
pub struct DormandPrince {
    /// Fixed value for the time step.
    dt: Scalar,
}

impl<Y, U> OdeIntegrator<Y, U> for DormandPrince
where
    Y: Tensor,
    U: TensorVec<Item = Y>,
{
}

impl<T> FixedStep<T> for DormandPrince {
    fn dt(&self) -> Quantity<T> {
        Quantity::new(self.dt)
    }
}

impl<Y, U, V, T> Explicit<Y, U, V, T> for DormandPrince
where
    Y: Differentiable<T> + Tensor,
    Derivative<Y, T>: Mul<Quantity<T>, Output = Y>,
    for<'a> &'a Y: Mul<Scalar, Output = Y> + Sub<&'a Y, Output = Y>,
    for<'a> &'a Derivative<Y, T>:
        Mul<Scalar, Output = Derivative<Y, T>> + Mul<Quantity<T>, Output = Y>,
    U: TensorVec<Item = Y>,
    V: TensorVec<Item = Derivative<Y, T>>,
{
    const SLOPES: usize = 6;
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

impl<Y, U, V, T> FixedStepExplicit<Y, U, V, T> for DormandPrince
where
    Y: Differentiable<T> + Tensor,
    Derivative<Y, T>: Mul<Quantity<T>, Output = Y>,
    for<'a> &'a Y: Mul<Scalar, Output = Y> + Sub<&'a Y, Output = Y>,
    for<'a> &'a Derivative<Y, T>:
        Mul<Scalar, Output = Derivative<Y, T>> + Mul<Quantity<T>, Output = Y>,
    U: TensorVec<Item = Y>,
    V: TensorVec<Item = Derivative<Y, T>>,
{
    type Tableau = DormandPrinceTableau;
}
