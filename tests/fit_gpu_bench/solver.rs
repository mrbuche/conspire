//! The limited-memory BFGS of the fit, written once for a backend that has the vectors.
//!
//! The method is the one of the fit of the library: a history of eight corrections, a first step
//! of a length of the mean size of the nodes, backtracking from a whole step by halves along the
//! two-loop direction until the energy falls by a part of what the slope promises, and a
//! correction kept when it has curvature.
//!
//! A backend is where the vectors are and how the work on them is done. The host can run it on
//! the arithmetic of the kernels, one thread after another ([`Emulated`]), and a device runs it
//! on the kernels. A third way, [`reference`], is written on its own with plain loops, to check
//! the other two against.
//!
//! The numbers that the two-loop recursion makes, the dot products against the vectors of the
//! history and the coefficients made of them, stay on the device, so the recursion needs no
//! round trip to the host. The host reads back what it decides on: the lengths of the gradient and
//! of the iterate, the slope, the energy of a trial point, and the curvature of a correction.

use super::{
    GROUP, Layout, Point, Problem, SUMMERS, gradient_on_host, objective_on_host, planes, points_of,
    shared,
};
use std::{cell::RefCell, collections::VecDeque, mem::swap};

/// The corrections kept.
pub const HISTORY: usize = 8;
/// How many numbers a backend keeps for the solver to read.
pub const SLOTS: usize = 8;
const BACKTRACKS: usize = 32;
const ARMIJO: f64 = 1.0e-4;
const CURVATURE_FLOOR: f64 = 1.0e-12;

/// The parts of a step, which a backend can time.
#[derive(Clone, Copy)]
pub enum Phase {
    Norms,
    TwoLoop,
    Slope,
    Search,
    Gradient,
    History,
}

#[cfg_attr(not(feature = "gpu"), allow(dead_code))]
pub const PHASES: usize = 6;

#[cfg_attr(not(feature = "gpu"), allow(dead_code))]
impl Phase {
    pub const ALL: [Self; PHASES] = [
        Self::Norms,
        Self::TwoLoop,
        Self::Slope,
        Self::Search,
        Self::Gradient,
        Self::History,
    ];
    pub fn name(self) -> &'static str {
        match self {
            Self::Norms => "norms",
            Self::TwoLoop => "two-loop",
            Self::Slope => "slope",
            Self::Search => "line search",
            Self::Gradient => "gradient",
            Self::History => "history",
        }
    }
}

#[derive(Clone, Copy)]
pub struct Settings {
    /// The most steps to take.
    pub steps: usize,
    /// The length of the gradient over the length of the iterate, at which to stop.
    pub convergence: f64,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            steps: 100,
            convergence: 1.0e-5,
        }
    }
}

/// What a run did.
#[derive(Debug, Default)]
pub struct Trace {
    /// The energy at the start.
    pub initial: f64,
    /// The energy after each step that was taken.
    pub values: Vec<f64>,
    /// The length of each step that was taken, in the length of the direction.
    pub steps: Vec<f64>,
    /// The passes of the loop that were made, those that took a step and those that did not.
    pub passes: usize,
    /// The times the energy was found.
    pub evaluations: usize,
    pub converged: bool,
}

/// The vectors of the solver, and the work on them.
///
/// A vector is a point of the grid on planes, as the kernels take them. The work is not waited
/// for, apart from what reads a number back.
pub trait Backend {
    type Vector;
    fn count(&self) -> usize;
    fn vector(&self) -> Self::Vector;
    fn upload(&self, vector: &Self::Vector, planes: &[f64]);
    fn download(&self, vector: &Self::Vector) -> Vec<f64>;
    /// The energy at `x`, which is read back.
    fn objective(&self, x: &Self::Vector) -> f64;
    /// The gradient at `x`, into `gradient`.
    fn gradient(&self, x: &Self::Vector, gradient: &Self::Vector);
    /// `out = y + alpha * x`.
    fn axpy(&self, alpha: f64, x: &Self::Vector, y: &Self::Vector, out: &Self::Vector);
    /// `out = alpha * x`.
    fn scale(&self, alpha: f64, x: &Self::Vector, out: &Self::Vector);
    /// `y *= alpha`.
    fn scale_in_place(&self, alpha: f64, y: &Self::Vector);
    /// The dot product of `a` and `b`, into the number `slot`.
    fn dot(&self, a: &Self::Vector, b: &Self::Vector, slot: usize);
    /// The numbers that the dot products made, which waits for them.
    fn read(&self) -> [f64; SLOTS];
    /// The number `index` of the alphas, or of the betas, set to `factor` times the number `slot`.
    fn times(&self, factor: f64, slot: usize, beta: bool, index: usize);
    /// `y -= alphas[index] * x`.
    fn subtract_alpha(&self, index: usize, x: &Self::Vector, y: &Self::Vector);
    /// `y -= (alphas[index] + betas[index]) * x`.
    fn subtract_pair(&self, index: usize, x: &Self::Vector, y: &Self::Vector);
    fn begin(&self, _phase: Phase) {}
    fn end(&self, _phase: Phase) {}
}

// The numbers that the solver has the backend keep, which are read together.
const GG: usize = 0;
const XX: usize = 1;
const GD: usize = 2;
const SY: usize = 3;
const SS: usize = 4;
const YY: usize = 5;
const TEMPORARY: usize = 6;

/// Fits from the point `x`, which is where it ends, with the sizes of the nodes of mean `typical`.
pub fn lbfgs<B: Backend>(
    backend: &B,
    x: &mut B::Vector,
    typical: f64,
    settings: Settings,
) -> Trace {
    let b = backend;
    let mut trace = Trace::default();
    let (mut g, mut g_new, d, mut trial) = (b.vector(), b.vector(), b.vector(), b.vector());
    // A correction is two vectors. There is one more than are kept, to write the next one in.
    let pool: Vec<(B::Vector, B::Vector)> =
        (0..=HISTORY).map(|_| (b.vector(), b.vector())).collect();
    let mut free: Vec<usize> = (0..=HISTORY).collect();
    let mut order: VecDeque<usize> = VecDeque::new();
    let (mut rho, mut gamma) = ([0.0; HISTORY + 1], [0.0; HISTORY + 1]);
    b.gradient(x, &g);
    let mut value = b.objective(x);
    trace.initial = value;
    trace.evaluations = 1;
    for _ in 0..settings.steps {
        b.begin(Phase::Norms);
        b.dot(&g, &g, GG);
        b.dot(x, x, XX);
        let read = b.read();
        b.end(Phase::Norms);
        let magnitude = read[GG].sqrt();
        if magnitude / read[XX].sqrt().max(1.0) < settings.convergence {
            trace.converged = true;
            break;
        }
        trace.passes += 1;
        b.begin(Phase::TwoLoop);
        b.scale(1.0, &g, &d);
        for &k in order.iter().rev() {
            b.dot(&pool[k].0, &d, TEMPORARY);
            b.times(rho[k], TEMPORARY, false, k);
            b.subtract_alpha(k, &pool[k].1, &d);
        }
        let initial = order.back().map_or(typical / magnitude, |&k| gamma[k]);
        b.scale_in_place(-initial, &d);
        for &k in &order {
            b.dot(&pool[k].1, &d, TEMPORARY);
            b.times(rho[k], TEMPORARY, true, k);
            b.subtract_pair(k, &pool[k].0, &d);
        }
        b.end(Phase::TwoLoop);
        b.begin(Phase::Slope);
        b.dot(&g, &d, GD);
        let slope = b.read()[GD];
        b.end(Phase::Slope);
        if slope >= 0.0 {
            free.extend(order.drain(..));
            continue;
        }
        b.begin(Phase::Search);
        let mut step = 1.0;
        let mut accepted = None;
        for _ in 0..BACKTRACKS {
            b.axpy(step, &d, x, &trial);
            let found = b.objective(&trial);
            trace.evaluations += 1;
            if found <= value + ARMIJO * step * slope {
                accepted = Some(found);
                break;
            }
            step *= 0.5;
        }
        b.end(Phase::Search);
        let Some(found) = accepted else {
            if order.is_empty() {
                break;
            }
            free.extend(order.drain(..));
            continue;
        };
        let slot = free.pop().expect("there is a correction to write in");
        b.begin(Phase::Gradient);
        b.scale(step, &d, &pool[slot].0);
        swap(x, &mut trial);
        b.gradient(x, &g_new);
        b.axpy(-1.0, &g, &g_new, &pool[slot].1);
        b.end(Phase::Gradient);
        b.begin(Phase::History);
        b.dot(&pool[slot].0, &pool[slot].1, SY);
        b.dot(&pool[slot].0, &pool[slot].0, SS);
        b.dot(&pool[slot].1, &pool[slot].1, YY);
        let read = b.read();
        b.end(Phase::History);
        let (sy, ss, yy) = (read[SY], read[SS], read[YY]);
        if sy > CURVATURE_FLOOR * ss.sqrt() * yy.sqrt() {
            if order.len() == HISTORY {
                free.extend(order.pop_front());
            }
            rho[slot] = 1.0 / sy;
            gamma[slot] = sy / yy;
            order.push_back(slot);
        } else {
            free.push(slot);
        }
        swap(&mut g, &mut g_new);
        value = found;
        trace.values.push(value);
        trace.steps.push(step);
    }
    trace
}

/// The same method with plain loops on the points, which uses the host reference for the energy.
pub fn reference(
    problem: &Problem,
    start: &[Point],
    typical: f64,
    settings: Settings,
) -> (Trace, Vec<Point>) {
    let count = 3 * problem.nodes();
    let mut x = planes(start);
    let objective = |x: &[f64]| problem.objective(&points_of(x));
    let gradient = |x: &[f64]| planes(&problem.gradient(&points_of(x)));
    let dot = |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(a, b)| a * b).sum::<f64>();
    let mut g = gradient(&x);
    let mut value = objective(&x);
    let mut trace = Trace {
        initial: value,
        evaluations: 1,
        ..Default::default()
    };
    // The correction, its change of the gradient, one over their product, and the scale of the
    // direction that it gives.
    let mut history: VecDeque<(Vec<f64>, Vec<f64>, f64, f64)> = VecDeque::new();
    for _ in 0..settings.steps {
        let magnitude = dot(&g, &g).sqrt();
        if magnitude / dot(&x, &x).sqrt().max(1.0) < settings.convergence {
            trace.converged = true;
            break;
        }
        trace.passes += 1;
        let mut d = g.clone();
        let mut alphas = vec![0.0; history.len()];
        for (k, (s, y, rho, _)) in history.iter().enumerate().rev() {
            alphas[k] = rho * dot(s, &d);
            d.iter_mut().zip(y).for_each(|(d, y)| *d -= alphas[k] * y);
        }
        let initial = history.back().map_or(typical / magnitude, |entry| entry.3);
        d.iter_mut().for_each(|d| *d *= -initial);
        for (k, (s, y, rho, _)) in history.iter().enumerate() {
            let beta = rho * dot(y, &d);
            d.iter_mut()
                .zip(s)
                .for_each(|(d, s)| *d -= (alphas[k] + beta) * s);
        }
        let slope = dot(&g, &d);
        if slope >= 0.0 {
            history.clear();
            continue;
        }
        let mut step = 1.0;
        let mut accepted = None;
        let mut trial = vec![0.0; count];
        for _ in 0..BACKTRACKS {
            trial
                .iter_mut()
                .zip(x.iter().zip(&d))
                .for_each(|(trial, (x, d))| *trial = x + step * d);
            let found = objective(&trial);
            trace.evaluations += 1;
            if found <= value + ARMIJO * step * slope {
                accepted = Some(found);
                break;
            }
            step *= 0.5;
        }
        let Some(found) = accepted else {
            if history.is_empty() {
                break;
            }
            history.clear();
            continue;
        };
        let s: Vec<f64> = d.iter().map(|d| step * d).collect();
        x = trial;
        let g_new = gradient(&x);
        let y: Vec<f64> = g_new.iter().zip(&g).map(|(new, old)| new - old).collect();
        let (sy, ss, yy) = (dot(&s, &y), dot(&s, &s), dot(&y, &y));
        if sy > CURVATURE_FLOOR * ss.sqrt() * yy.sqrt() {
            if history.len() == HISTORY {
                history.pop_front();
            }
            history.push_back((s, y, 1.0 / sy, sy / yy));
        }
        g = g_new;
        value = found;
        trace.values.push(value);
        trace.steps.push(step);
    }
    (trace, points_of(&x))
}

/// The kernels run on the host, one thread after another.
pub struct Emulated<'a> {
    layout: &'a Layout,
    dots: RefCell<[f64; SLOTS]>,
    alphas: RefCell<[f64; HISTORY + 1]>,
    betas: RefCell<[f64; HISTORY + 1]>,
}

impl<'a> Emulated<'a> {
    pub fn new(layout: &'a Layout) -> Self {
        Self {
            layout,
            dots: RefCell::new([0.0; SLOTS]),
            alphas: RefCell::new([0.0; HISTORY + 1]),
            betas: RefCell::new([0.0; HISTORY + 1]),
        }
    }
}

impl Backend for Emulated<'_> {
    type Vector = RefCell<Vec<f64>>;
    fn count(&self) -> usize {
        3 * self.layout.nodes
    }
    fn vector(&self) -> Self::Vector {
        RefCell::new(vec![0.0; self.count()])
    }
    fn upload(&self, vector: &Self::Vector, planes: &[f64]) {
        vector.borrow_mut().copy_from_slice(planes);
    }
    fn download(&self, vector: &Self::Vector) -> Vec<f64> {
        vector.borrow().clone()
    }
    fn objective(&self, x: &Self::Vector) -> f64 {
        objective_on_host(self.layout, &x.borrow(), SUMMERS)
    }
    fn gradient(&self, x: &Self::Vector, gradient: &Self::Vector) {
        *gradient.borrow_mut() = gradient_on_host(self.layout, &x.borrow());
    }
    fn axpy(&self, alpha: f64, x: &Self::Vector, y: &Self::Vector, out: &Self::Vector) {
        let (x, y, mut out) = (x.borrow(), y.borrow(), out.borrow_mut());
        for item in 0..x.len() {
            // SAFETY: the vectors are `x.len()` values.
            unsafe {
                shared::axpy(
                    item,
                    x.len(),
                    alpha,
                    x.as_ptr(),
                    y.as_ptr(),
                    out.as_mut_ptr(),
                )
            }
        }
    }
    fn scale(&self, alpha: f64, x: &Self::Vector, out: &Self::Vector) {
        let (x, mut out) = (x.borrow(), out.borrow_mut());
        for item in 0..x.len() {
            // SAFETY: the vectors are `x.len()` values.
            unsafe { shared::scale(item, x.len(), alpha, x.as_ptr(), out.as_mut_ptr()) }
        }
    }
    fn scale_in_place(&self, alpha: f64, y: &Self::Vector) {
        let mut y = y.borrow_mut();
        for item in 0..y.len() {
            // SAFETY: the vector is `y.len()` values.
            unsafe { shared::scale_in_place(item, y.len(), alpha, y.as_mut_ptr()) }
        }
    }
    fn dot(&self, a: &Self::Vector, b: &Self::Vector, slot: usize) {
        let (a, b) = (a.borrow(), b.borrow());
        let mut partials = vec![0.0; SUMMERS];
        for item in 0..SUMMERS {
            // SAFETY: the vectors are `a.len()` values and the partials are `SUMMERS`.
            unsafe {
                shared::dot_strided(
                    item,
                    a.len(),
                    SUMMERS,
                    a.as_ptr(),
                    b.as_ptr(),
                    partials.as_mut_ptr(),
                )
            }
        }
        let mut groups = vec![0.0; SUMMERS.div_ceil(GROUP)];
        for item in 0..groups.len() {
            // SAFETY: the partials are `SUMMERS` and the groups are as many as that over `GROUP`.
            unsafe {
                shared::sum_groups(item, SUMMERS, GROUP, partials.as_ptr(), groups.as_mut_ptr())
            }
        }
        let mut dots = self.dots.borrow_mut();
        // SAFETY: the groups are as many as they are, and the slot is one of the numbers.
        unsafe { shared::sum_into(0, groups.len(), groups.as_ptr(), dots.as_mut_ptr(), slot) }
    }
    fn read(&self) -> [f64; SLOTS] {
        *self.dots.borrow()
    }
    fn times(&self, factor: f64, slot: usize, beta: bool, index: usize) {
        let dots = self.dots.borrow();
        let mut out = if beta {
            self.betas.borrow_mut()
        } else {
            self.alphas.borrow_mut()
        };
        // SAFETY: the slot and the index are among the numbers.
        unsafe { shared::times(0, factor, dots.as_ptr(), slot, out.as_mut_ptr(), index) }
    }
    fn subtract_alpha(&self, index: usize, x: &Self::Vector, y: &Self::Vector) {
        let (alphas, x, mut y) = (self.alphas.borrow(), x.borrow(), y.borrow_mut());
        for item in 0..x.len() {
            // SAFETY: the vectors are `x.len()` values, and the index is one of the alphas.
            unsafe {
                shared::subtract_alpha(
                    item,
                    x.len(),
                    alphas.as_ptr(),
                    index,
                    x.as_ptr(),
                    y.as_mut_ptr(),
                )
            }
        }
    }
    fn subtract_pair(&self, index: usize, x: &Self::Vector, y: &Self::Vector) {
        let (alphas, betas) = (self.alphas.borrow(), self.betas.borrow());
        let (x, mut y) = (x.borrow(), y.borrow_mut());
        for item in 0..x.len() {
            // SAFETY: the vectors are `x.len()` values, and the index is one of the numbers.
            unsafe {
                shared::subtract_pair(
                    item,
                    x.len(),
                    alphas.as_ptr(),
                    betas.as_ptr(),
                    index,
                    x.as_ptr(),
                    y.as_mut_ptr(),
                )
            }
        }
    }
}

/// How far two runs are from each other: the greatest difference of the energies they found over
/// the energy, and whether the lengths of the steps are the same.
pub fn compare(a: &Trace, b: &Trace) -> (f64, bool) {
    let difference = a
        .values
        .iter()
        .zip(&b.values)
        .map(|(a, b)| (a - b).abs() / b.abs())
        .fold(0.0, f64::max);
    (
        difference,
        a.steps == b.steps && a.values.len() == b.values.len(),
    )
}

#[test]
fn the_solver_on_the_kernels_follows_the_reference() {
    let problem = Problem::new(6, 1.0);
    let layout = Layout::new(&problem);
    let typical = problem.lengths.iter().sum::<f64>() / problem.nodes() as f64;
    let settings = Settings {
        steps: 30,
        ..Default::default()
    };
    let (expected, expected_x) = reference(&problem, &problem.coordinates, typical, settings);
    let backend = Emulated::new(&layout);
    let mut x = backend.vector();
    backend.upload(&x, &planes(&problem.coordinates));
    let trace = lbfgs(&backend, &mut x, typical, settings);
    // It takes steps, and what it finds is lower.
    assert!(
        expected.values.len() >= 20,
        "{} steps",
        expected.values.len()
    );
    assert!(expected.values[expected.values.len() - 1] < 0.5 * expected.initial);
    let (difference, same_steps) = compare(&trace, &expected);
    assert!(same_steps, "{:?} against {:?}", trace.steps, expected.steps);
    assert!(difference < 1.0e-9, "the energies differ by {difference}");
    assert_eq!(trace.evaluations, expected.evaluations);
    let error = super::disagreement(&expected_x, &points_of(&backend.download(&x)));
    assert!(error < 1.0e-8, "the points differ by {error}");
}
