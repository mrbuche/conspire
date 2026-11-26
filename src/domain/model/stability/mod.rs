//! Whether a run of explicit integration checks that its time step is stable.

use crate::math::Scalar;

/// Takes the steps that are given, however large they are.
#[derive(Clone, Copy, Debug, Default)]
pub struct Unchecked;

/// Checks each step against the fastest time scale of the elements, as `estimate` finds it.
///
/// A fixed time step above the fraction `safety` of the stability limit is an error, and a
/// variable time step is reduced to it. The estimate is refreshed every `interval` evaluations
/// of the bound, which for the integrators here is every step that is tried. It is that of the
/// lumped masses and heat capacities, which consistent ones exceed, so only a run with the
/// lumped ones can be checked.
#[derive(Clone, Copy, Debug)]
pub struct Checked<E> {
    pub estimate: E,
    pub interval: usize,
    pub safety: Scalar,
}

/// Estimates the time scales by power iteration on the tangent of each element, or of each node
/// and its neighbors where there are no elements, and takes the shortest.
#[derive(Clone, Copy, Debug, Default)]
pub struct ElementPowerIteration;
