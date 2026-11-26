//! Whether a run of explicit integration checks that its time step is stable.

use crate::{domain::ModelBound, math::Scalar};

/// Takes the steps that are given, however large they are.
#[derive(Clone, Copy, Debug, Default)]
pub struct Unchecked;

/// Checks each step against the fastest time scale of the elements, as `estimate` finds it.
///
/// A fixed time step above the fraction `safety` of the stability limit is an error, and a
/// variable time step is reduced to it. The estimate is refreshed every `interval` evaluations
/// of the bound, which for the integrators here is every step that is tried, and the steps in
/// between are checked against the last estimate. A limit that falls is then noticed up to
/// `interval - 1` steps late, or not at all if the run ends first. The estimate is that of the
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

/// Bounds the highest angular frequency of each element from its gradient vectors, its lumped
/// masses, and the stiffness at its integration points as `stiffness` has it, and takes the
/// shortest time scale.
///
/// It needs no stiffnesses of the elements and no iteration, so it costs much less than
/// [`ElementPowerIteration`], and its time scales are no longer than those of the elements.
#[derive(Clone, Copy, Debug)]
pub struct CourantFriedrichsLewy<S = ModelBound> {
    pub stiffness: S,
}

impl Default for CourantFriedrichsLewy {
    fn default() -> Self {
        Self {
            stiffness: ModelBound,
        }
    }
}

impl Default for Checked<CourantFriedrichsLewy> {
    fn default() -> Self {
        Self {
            estimate: CourantFriedrichsLewy::default(),
            interval: 10,
            safety: 0.9,
        }
    }
}
