#[cfg(test)]
mod test;

use crate::{domain::boundary::NodalState, math::Quantity, units::Time};
use core::{marker::PhantomData, ops::Add};

/// An external force that a device subtracts from the forces of a model, which does not depend on
/// the state of the body, as plain numbers.
#[cfg(feature = "gpu")]
#[derive(Clone, Debug, PartialEq)]
pub enum DeviceForce {
    /// A force on each degree of freedom, as `3 * node + component`, which does not change.
    Constant(Vec<f64>),
    /// The weight of each node of a force on the nodes that is a function of time, which is the
    /// weight times three components that [`DeviceLoads::device_components`] gives at a time.
    Weighted(Vec<f64>),
}

#[cfg(feature = "gpu")]
impl DeviceForce {
    /// That the force has a value for each degree of freedom, or a weight for each node, of a
    /// model of `nodes` nodes.
    pub fn check(&self, nodes: usize) -> Result<(), String> {
        match self {
            Self::Constant(forces) if forces.len() != 3 * nodes => Err(format!(
                "a load has {} values, and the model has {} degrees of freedom",
                forces.len(),
                3 * nodes
            )),
            Self::Weighted(weights) if weights.len() != nodes => Err(format!(
                "a load has {} weights, and the model has {nodes} nodes",
                weights.len()
            )),
            _ => Ok(()),
        }
    }
}

/// External forces that a device subtracts from the forces of a model, in the order that they are
/// subtracted in, which matters to the last bit.
///
/// These are the forces that do not depend on the state of the body. A force that follows the
/// body, such as a pressure that follows the surface it acts on, is not one of them, and neither
/// is a list of forces that has one, so that to run such a list on a device is an error when the
/// program is built and not a wrong answer when it runs.
#[cfg(feature = "gpu")]
pub trait DeviceLoads {
    /// Gives each of the forces to `each`, in the order that they are subtracted in.
    fn device_forces(&self, each: &mut impl FnMut(DeviceForce));
    /// How many of the forces are weighted, which are a function of time.
    fn device_weighted(&self) -> usize;
    /// The three components, at a time, of the weighted force that is the `which`-th of those that
    /// [`device_forces`](Self::device_forces) gives, counting only the weighted ones.
    ///
    /// # Panics
    ///
    /// If there is no such weighted force, as `which` is not less than
    /// [`device_weighted`](Self::device_weighted).
    fn device_components(&self, which: usize, time: Quantity<Time>) -> [f64; 3];
    /// The components of the weighted force `which` at each of the times, three to a time, one
    /// time after another.
    fn device_component_rows(&self, which: usize, times: &[Quantity<Time>]) -> Vec<f64> {
        times
            .iter()
            .flat_map(|&time| self.device_components(which, time))
            .collect()
    }
}

/// External forces on the degrees of freedom of a state, which may be a function of time.
pub trait LoadAt<X, R> {
    /// Subtracts the external forces at a time and a state from a residual.
    fn subtract_at(&self, time: Quantity<Time>, state: &X, residual: &mut R) -> Result<(), String>;
}

/// External forces on the degrees of freedom of a state, which a residual of type `R` takes up.
///
/// A solve that is not a function of time takes these, so loads that are a function of time are
/// not accepted by it.
pub trait Load<X, R> {
    /// Subtracts the external forces at a state from a residual.
    fn subtract_from(&self, state: &X, residual: &mut R) -> Result<(), String>;
}

/// The stiffness of external forces that depend on the state, such as a pressure that follows the
/// surface it acts on, which a tangent of type `J` takes up.
///
/// A load that does not depend on the state has none to add, so the default adds nothing.
pub trait LoadStiffness<X, J> {
    /// Adds the derivative of the residual of the external forces to a tangent.
    fn add_stiffness_to(&self, _state: &X, _tangent: &mut J) -> Result<(), String> {
        Ok(())
    }
}

/// The potential energy of external forces that have one, which may be a function of time.
pub trait ConservativeAt<X, E> {
    /// The potential energy of the external forces at a time and a state.
    fn potential_energy_at(&self, time: Quantity<Time>, state: &X) -> Result<E, String>;
}

/// The potential energy of external forces that have one, so that a minimization can take them up.
pub trait Conservative<X, E>: ConservativeAt<X, E> {
    /// The potential energy of the external forces at a state.
    fn potential_energy(&self, state: &X) -> Result<E, String>;
}

/// The power of external forces, which a minimization over the rates of a state takes up.
///
/// The power is linear in the rates, so loads that are not conservative have one.
pub trait PowerAt<X, V, P> {
    /// The power of the external forces at a time and a state, through rates.
    fn power_at(&self, time: Quantity<Time>, state: &X, rates: &V) -> Result<P, String>;
}

/// External loads on the degrees of freedom of a layout, held in a nest of pairs.
///
/// The loads a layout can have are the builders defined for it, so that a load cannot be given
/// to a solve of another layout.
#[derive(Clone, Debug)]
pub struct Loads<L, T = ()>(T, PhantomData<L>);

impl<L> Loads<L> {
    pub fn new() -> Self {
        Self((), PhantomData)
    }
}

impl<L> Default for Loads<L> {
    fn default() -> Self {
        Self::new()
    }
}

impl<L, T> Loads<L, T> {
    /// Adds a load.
    pub fn with<A>(self, load: A) -> Loads<L, (T, A)> {
        Loads((self.0, load), PhantomData)
    }
}

impl<X, R> LoadAt<X, R> for () {
    fn subtract_at(&self, _: Quantity<Time>, _: &X, _: &mut R) -> Result<(), String> {
        Ok(())
    }
}

#[cfg(feature = "gpu")]
impl DeviceLoads for () {
    fn device_forces(&self, _: &mut impl FnMut(DeviceForce)) {}
    fn device_weighted(&self) -> usize {
        0
    }
    fn device_components(&self, which: usize, _: Quantity<Time>) -> [f64; 3] {
        panic!("there is no force of time {which}, as there are no forces")
    }
}

impl<X, R> Load<X, R> for () {
    fn subtract_from(&self, _: &X, _: &mut R) -> Result<(), String> {
        Ok(())
    }
}

/// The stiffness of external forces that depend on the state and may be a function of time.
///
/// A load that does not depend on the state has none to add, so the default adds nothing.
pub trait LoadStiffnessAt<X, J> {
    /// Adds the derivative of the residual of the external forces at a time to a tangent.
    fn add_stiffness_at(
        &self,
        _time: Quantity<Time>,
        _state: &X,
        _tangent: &mut J,
    ) -> Result<(), String> {
        Ok(())
    }
}

impl<X, J> LoadStiffness<X, J> for () {}

impl<X, J> LoadStiffnessAt<X, J> for () {}

impl<X, J, A, B> LoadStiffnessAt<X, J> for (A, B)
where
    A: LoadStiffnessAt<X, J>,
    B: LoadStiffnessAt<X, J>,
{
    fn add_stiffness_at(
        &self,
        time: Quantity<Time>,
        state: &X,
        tangent: &mut J,
    ) -> Result<(), String> {
        self.0.add_stiffness_at(time, state, tangent)?;
        self.1.add_stiffness_at(time, state, tangent)
    }
}

impl<L, T, X, J> LoadStiffnessAt<X, J> for Loads<L, T>
where
    X: NodalState<Layout = L>,
    T: LoadStiffnessAt<X, J>,
{
    fn add_stiffness_at(
        &self,
        time: Quantity<Time>,
        state: &X,
        tangent: &mut J,
    ) -> Result<(), String> {
        self.0.add_stiffness_at(time, state, tangent)
    }
}

impl<X, J, A, B> LoadStiffness<X, J> for (A, B)
where
    A: LoadStiffness<X, J>,
    B: LoadStiffness<X, J>,
{
    fn add_stiffness_to(&self, state: &X, tangent: &mut J) -> Result<(), String> {
        self.0.add_stiffness_to(state, tangent)?;
        self.1.add_stiffness_to(state, tangent)
    }
}

impl<L, T, X, J> LoadStiffness<X, J> for Loads<L, T>
where
    X: NodalState<Layout = L>,
    T: LoadStiffness<X, J>,
{
    fn add_stiffness_to(&self, state: &X, tangent: &mut J) -> Result<(), String> {
        self.0.add_stiffness_to(state, tangent)
    }
}

impl<X, E> ConservativeAt<X, E> for ()
where
    E: Default,
{
    fn potential_energy_at(&self, _: Quantity<Time>, _: &X) -> Result<E, String> {
        Ok(E::default())
    }
}

impl<X, E> Conservative<X, E> for ()
where
    E: Default,
{
    fn potential_energy(&self, _: &X) -> Result<E, String> {
        Ok(E::default())
    }
}

impl<X, V, P> PowerAt<X, V, P> for ()
where
    P: Default,
{
    fn power_at(&self, _: Quantity<Time>, _: &X, _: &V) -> Result<P, String> {
        Ok(P::default())
    }
}

impl<X, V, P, A, B> PowerAt<X, V, P> for (A, B)
where
    A: PowerAt<X, V, P>,
    B: PowerAt<X, V, P>,
    P: Add<Output = P>,
{
    fn power_at(&self, time: Quantity<Time>, state: &X, rates: &V) -> Result<P, String> {
        Ok(self.0.power_at(time, state, rates)? + self.1.power_at(time, state, rates)?)
    }
}

impl<L, T, X, V, P> PowerAt<X, V, P> for Loads<L, T>
where
    X: NodalState<Layout = L>,
    T: PowerAt<X, V, P>,
{
    fn power_at(&self, time: Quantity<Time>, state: &X, rates: &V) -> Result<P, String> {
        self.0.power_at(time, state, rates)
    }
}

impl<X, R, A, B> LoadAt<X, R> for (A, B)
where
    A: LoadAt<X, R>,
    B: LoadAt<X, R>,
{
    fn subtract_at(&self, time: Quantity<Time>, state: &X, residual: &mut R) -> Result<(), String> {
        self.0.subtract_at(time, state, residual)?;
        self.1.subtract_at(time, state, residual)
    }
}

#[cfg(feature = "gpu")]
impl<A, B> DeviceLoads for (A, B)
where
    A: DeviceLoads,
    B: DeviceLoads,
{
    fn device_forces(&self, each: &mut impl FnMut(DeviceForce)) {
        self.0.device_forces(each);
        self.1.device_forces(each);
    }
    fn device_weighted(&self) -> usize {
        self.0.device_weighted() + self.1.device_weighted()
    }
    fn device_components(&self, which: usize, time: Quantity<Time>) -> [f64; 3] {
        let before = self.0.device_weighted();
        if which < before {
            self.0.device_components(which, time)
        } else {
            self.1.device_components(which - before, time)
        }
    }
}

impl<X, R, A, B> Load<X, R> for (A, B)
where
    A: Load<X, R>,
    B: Load<X, R>,
{
    fn subtract_from(&self, state: &X, residual: &mut R) -> Result<(), String> {
        self.0.subtract_from(state, residual)?;
        self.1.subtract_from(state, residual)
    }
}

impl<X, E, A, B> ConservativeAt<X, E> for (A, B)
where
    A: ConservativeAt<X, E>,
    B: ConservativeAt<X, E>,
    E: Add<Output = E>,
{
    fn potential_energy_at(&self, time: Quantity<Time>, state: &X) -> Result<E, String> {
        Ok(self.0.potential_energy_at(time, state)? + self.1.potential_energy_at(time, state)?)
    }
}

impl<X, E, A, B> Conservative<X, E> for (A, B)
where
    A: Conservative<X, E>,
    B: Conservative<X, E>,
    E: Add<Output = E>,
{
    fn potential_energy(&self, state: &X) -> Result<E, String> {
        Ok(self.0.potential_energy(state)? + self.1.potential_energy(state)?)
    }
}

impl<L, T, X, R> LoadAt<X, R> for Loads<L, T>
where
    X: NodalState<Layout = L>,
    T: LoadAt<X, R>,
{
    fn subtract_at(&self, time: Quantity<Time>, state: &X, residual: &mut R) -> Result<(), String> {
        self.0.subtract_at(time, state, residual)
    }
}

#[cfg(feature = "gpu")]
impl<L, T> DeviceLoads for Loads<L, T>
where
    T: DeviceLoads,
{
    fn device_forces(&self, each: &mut impl FnMut(DeviceForce)) {
        self.0.device_forces(each);
    }
    fn device_weighted(&self) -> usize {
        self.0.device_weighted()
    }
    fn device_components(&self, which: usize, time: Quantity<Time>) -> [f64; 3] {
        self.0.device_components(which, time)
    }
}

impl<L, T, X, R> Load<X, R> for Loads<L, T>
where
    X: NodalState<Layout = L>,
    T: Load<X, R>,
{
    fn subtract_from(&self, state: &X, residual: &mut R) -> Result<(), String> {
        self.0.subtract_from(state, residual)
    }
}

impl<L, T, X, E> ConservativeAt<X, E> for Loads<L, T>
where
    X: NodalState<Layout = L>,
    T: ConservativeAt<X, E>,
{
    fn potential_energy_at(&self, time: Quantity<Time>, state: &X) -> Result<E, String> {
        self.0.potential_energy_at(time, state)
    }
}

impl<L, T, X, E> Conservative<X, E> for Loads<L, T>
where
    X: NodalState<Layout = L>,
    T: Conservative<X, E>,
{
    fn potential_energy(&self, state: &X) -> Result<E, String> {
        self.0.potential_energy(state)
    }
}
