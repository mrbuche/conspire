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
    GROUP, Layout, Point, Problem, SUMMERS, chi, dot, gradient_on_host, min_by_passes, minus,
    objective_on_host, planes, points_of, shared, sum_by_passes,
};
use std::{
    cell::{Cell, RefCell},
    collections::VecDeque,
    marker::PhantomData,
    mem::swap,
    thread::scope,
    time::Instant,
};

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
    /// The problem, with the sizes and the targets that a sweep sets.
    layout: RefCell<Layout>,
    marker: PhantomData<&'a Layout>,
    targets: Option<Targets<'a>>,
    threads: usize,
    group: usize,
    dots: RefCell<[f64; SLOTS]>,
    alphas: RefCell<[f64; HISTORY + 1]>,
    betas: RefCell<[f64; HISTORY + 1]>,
    batch: RefCell<[f64; BATCH]>,
}

impl<'a> Emulated<'a> {
    pub fn new(layout: &'a Layout) -> Self {
        Self::with_batch(layout, BATCH_THREADS, BATCH_GROUP)
    }
    /// With the targets of the faces found by `targets` from where the nodes are, in place of
    /// those of a sphere.
    #[cfg_attr(not(feature = "geometry"), allow(dead_code))]
    pub fn with_targets(mut self, targets: Targets<'a>) -> Self {
        self.targets = Some(targets);
        self
    }
    /// With other threads for the first pass of the sums of a batch, and other groups after it.
    pub fn with_batch(layout: &'a Layout, threads: usize, group: usize) -> Self {
        assert!(threads >= 2 && group >= 2);
        Self {
            layout: RefCell::new(layout.clone()),
            marker: PhantomData,
            targets: None,
            threads,
            group,
            dots: RefCell::new([0.0; SLOTS]),
            alphas: RefCell::new([0.0; HISTORY + 1]),
            betas: RefCell::new([0.0; HISTORY + 1]),
            batch: RefCell::new([0.0; BATCH]),
        }
    }
}

impl Backend for Emulated<'_> {
    type Vector = RefCell<Vec<f64>>;
    fn count(&self) -> usize {
        3 * self.layout.borrow().nodes
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
        objective_on_host(&self.layout.borrow(), &x.borrow(), self.threads, self.group)
    }
    fn gradient(&self, x: &Self::Vector, gradient: &Self::Vector) {
        *gradient.borrow_mut() = gradient_on_host(&self.layout.borrow(), &x.borrow());
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

// The compact form.
//
// The two-loop recursion takes a dot product, and with it a reduction, for each of the corrections
// twice over, which on a device is a chain of dependent launches. The compact form of the same
// inverse Hessian,
//
//     H = g I + [S gY] [[R^-T (D + g Y'Y) R^-1, -R^-T], [-R^-1, 0]] [S'; gY'],
//
// where `S` and `Y` hold the corrections, `R` is the upper triangle of `S'Y`, `D` is its diagonal
// and `g` is the scale of the newest correction, takes all of the dot products against the history
// in one pass over the vectors, a small triangular solve on the host, and the direction in one
// pass more. The slope comes from the same dot products, with none of its own.

/// The threads that add up the products of a batch of dot products, and how many partial sums a
/// thread of a later pass adds up, unless they are set otherwise. The passes take the threads to
/// groups, and the groups to groups, until there is one sum.
pub const BATCH_THREADS: usize = 32768;
pub const BATCH_GROUP: usize = 32;
/// The threads of the first pass of the sums of a batch that were fastest on an H100 for vectors
/// of `count` values: enough to keep the memory busy, and not so many that their partial sums, a
/// row of them for each dot product, are a part of what is read. For vectors of ten thousand values
/// to ten million and more it was 16384 to 131072 threads.
pub fn automatic_threads(count: usize) -> usize {
    (count / 16).next_power_of_two().clamp(16384, 131072)
}

/// The partial sums that a thread of the later passes adds up, for the threads of the first pass:
/// short chains for few threads, and few passes for many.
pub fn automatic_group(threads: usize) -> usize {
    if threads <= 32768 { 32 } else { 64 }
}

/// What finds the targets of the faces from where the nodes are: the points, the normals and the
/// squared distances of the faces, on planes, as the layout has them.
pub type Targets<'a> = Box<dyn Fn(&[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>) + 'a>;

/// The most dot products of a batch.
pub const BATCH: usize = shared::ROWS_UPDATE;
/// The dot products of the first pass of a step.
pub const ROWS_HISTORY: usize = shared::ROWS_HISTORY;
/// The dot products of the update of the history.
pub const ROWS_UPDATE: usize = shared::ROWS_UPDATE;

/// A backend that can take the dot products of a step in batches, and keeps the corrections in a
/// pool.
pub trait CompactBackend: Backend {
    type Pool;
    fn pool(&self) -> Self::Pool;
    /// The correction in the slot `slot` has the step `alpha * x`.
    fn store_step(&self, alpha: f64, x: &Self::Vector, pool: &Self::Pool, slot: usize);
    /// The correction in the slot `slot` has the change of the gradient `new - old`.
    fn store_change(&self, new: &Self::Vector, old: &Self::Vector, pool: &Self::Pool, slot: usize);
    /// The dot products of the first pass of a step, for the corrections in `order`.
    fn history_dots(&self, pool: &Self::Pool, order: &[usize], g: &Self::Vector, x: &Self::Vector);
    /// The dot products of the update, once the correction in the slot `slot` is written.
    fn update_dots(&self, pool: &Self::Pool, order: &[usize], slot: usize);
    /// The first `rows` dot products of the batch, which waits for them.
    fn read_batch(&self, rows: usize) -> Vec<f64>;
    /// `d` is the sum of the vectors with the coefficients that are given: the first for `g`, then
    /// those for the steps in `order`, then those for the changes of the gradient.
    fn combine(
        &self,
        pool: &Self::Pool,
        order: &[usize],
        coefficients: &[f64; 1 + 2 * HISTORY],
        g: &Self::Vector,
        d: &Self::Vector,
    );
}

/// The coefficients of the direction `-H g` on `g`, the steps and the changes of the gradient,
/// and the slope `g . (-H g)`, from the `m` corrections in order: the dot products of each step
/// with each change `sy[i][j]` and of the changes with each other `yy[i][j]`, those of the
/// steps and of the changes with the gradient `a` and `b`, the gradient with itself `gg`, and the
/// scale `gamma` of the initial inverse Hessian.
fn compact(
    m: usize,
    sy: &[[f64; HISTORY]; HISTORY],
    yy: &[[f64; HISTORY]; HISTORY],
    a: &[f64; HISTORY],
    b: &[f64; HISTORY],
    gamma: f64,
    gg: f64,
) -> ([f64; 1 + 2 * HISTORY], f64) {
    let mut coefficients = [0.0; 1 + 2 * HISTORY];
    coefficients[0] = -gamma;
    // `R w = a`, with `R` the upper triangle of `S'Y`.
    let mut w = [0.0; HISTORY];
    for i in (0..m).rev() {
        let mut sum = a[i];
        for j in i + 1..m {
            sum -= sy[i][j] * w[j];
        }
        w[i] = sum / sy[i][i];
    }
    // `R' z = (D + gamma Y'Y) w - gamma b`.
    let mut q = [0.0; HISTORY];
    for i in 0..m {
        let mut sum = sy[i][i] * w[i];
        for j in 0..m {
            sum += gamma * yy[i][j] * w[j];
        }
        q[i] = sum - gamma * b[i];
    }
    let mut z = [0.0; HISTORY];
    for i in 0..m {
        let mut sum = q[i];
        for j in 0..i {
            sum -= sy[j][i] * z[j];
        }
        z[i] = sum / sy[i][i];
    }
    // `-H g = -gamma g - S z + gamma Y w`.
    let mut slope = -gamma * gg;
    for j in 0..m {
        coefficients[1 + j] = -z[j];
        coefficients[1 + HISTORY + j] = gamma * w[j];
        slope += -z[j] * a[j] + gamma * w[j] * b[j];
    }
    (coefficients, slope)
}

/// Fits as [`lbfgs`] does, with the compact form of the inverse Hessian in place of the two-loop
/// recursion, which is the same method with other arithmetic.
pub fn lbfgs_compact<B: CompactBackend>(
    backend: &B,
    x: &mut B::Vector,
    typical: f64,
    settings: Settings,
) -> Trace {
    let b = backend;
    let mut trace = Trace::default();
    let (mut g, mut g_new, d, mut trial) = (b.vector(), b.vector(), b.vector(), b.vector());
    let pool = b.pool();
    let mut free: Vec<usize> = (0..=HISTORY).collect();
    let mut order: VecDeque<usize> = VecDeque::new();
    // The dot products of the steps with the changes of the gradient, and of the changes with
    // each other, by the slots of the corrections.
    let mut sy = [[0.0; HISTORY + 1]; HISTORY + 1];
    let mut yy = [[0.0; HISTORY + 1]; HISTORY + 1];
    b.gradient(x, &g);
    let mut value = b.objective(x);
    trace.initial = value;
    trace.evaluations = 1;
    for _ in 0..settings.steps {
        let slots: Vec<usize> = order.iter().copied().collect();
        let m = slots.len();
        b.begin(Phase::Norms);
        b.history_dots(&pool, &slots, &g, x);
        let read = b.read_batch(ROWS_HISTORY);
        b.end(Phase::Norms);
        let gg = read[0];
        let magnitude = gg.sqrt();
        if magnitude / read[1].sqrt().max(1.0) < settings.convergence {
            trace.converged = true;
            break;
        }
        trace.passes += 1;
        b.begin(Phase::TwoLoop);
        let gamma = slots
            .last()
            .map_or(typical / magnitude, |&k| sy[k][k] / yy[k][k]);
        let (mut a_in_order, mut b_in_order) = ([0.0; HISTORY], [0.0; HISTORY]);
        let (mut sy_in_order, mut yy_in_order) =
            ([[0.0; HISTORY]; HISTORY], [[0.0; HISTORY]; HISTORY]);
        for i in 0..m {
            a_in_order[i] = read[2 + i];
            b_in_order[i] = read[2 + HISTORY + i];
            for j in 0..m {
                sy_in_order[i][j] = sy[slots[i]][slots[j]];
                yy_in_order[i][j] = yy[slots[i]][slots[j]];
            }
        }
        let (coefficients, slope) = compact(
            m,
            &sy_in_order,
            &yy_in_order,
            &a_in_order,
            &b_in_order,
            gamma,
            gg,
        );
        b.combine(&pool, &slots, &coefficients, &g, &d);
        b.end(Phase::TwoLoop);
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
        b.store_step(step, &d, &pool, slot);
        swap(x, &mut trial);
        b.gradient(x, &g_new);
        b.store_change(&g_new, &g, &pool, slot);
        b.end(Phase::Gradient);
        b.begin(Phase::History);
        b.update_dots(&pool, &slots, slot);
        let read = b.read_batch(ROWS_UPDATE);
        b.end(Phase::History);
        let (new_sy, new_ss, new_yy) = (read[0], read[1], read[2]);
        if new_sy > CURVATURE_FLOOR * new_ss.sqrt() * new_yy.sqrt() {
            for (i, &old) in slots.iter().enumerate() {
                sy[old][slot] = read[3 + i];
                sy[slot][old] = read[3 + HISTORY + i];
                yy[old][slot] = read[3 + 2 * HISTORY + i];
                yy[slot][old] = read[3 + 2 * HISTORY + i];
            }
            sy[slot][slot] = new_sy;
            yy[slot][slot] = new_yy;
            if order.len() == HISTORY {
                free.extend(order.pop_front());
            }
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

/// The corrections of the host: two pools of vectors one after another.
pub struct EmulatedPool {
    s: RefCell<Vec<f64>>,
    y: RefCell<Vec<f64>>,
}

impl CompactBackend for Emulated<'_> {
    type Pool = EmulatedPool;
    fn pool(&self) -> EmulatedPool {
        EmulatedPool {
            s: RefCell::new(vec![0.0; (HISTORY + 1) * self.count()]),
            y: RefCell::new(vec![0.0; (HISTORY + 1) * self.count()]),
        }
    }
    fn store_step(&self, alpha: f64, x: &Self::Vector, pool: &EmulatedPool, slot: usize) {
        let count = self.count();
        let (x, mut s) = (x.borrow(), pool.s.borrow_mut());
        for item in 0..count {
            // SAFETY: the vector is `count` values and the pool has the slot.
            unsafe {
                shared::scale_at(item, count, alpha, x.as_ptr(), s.as_mut_ptr(), slot * count)
            }
        }
    }
    fn store_change(
        &self,
        new: &Self::Vector,
        old: &Self::Vector,
        pool: &EmulatedPool,
        slot: usize,
    ) {
        let count = self.count();
        let (new, old, mut y) = (new.borrow(), old.borrow(), pool.y.borrow_mut());
        for item in 0..count {
            // SAFETY: the vectors are `count` values and the pool has the slot.
            unsafe {
                shared::difference_at(
                    item,
                    count,
                    new.as_ptr(),
                    old.as_ptr(),
                    y.as_mut_ptr(),
                    slot * count,
                )
            }
        }
    }
    fn history_dots(
        &self,
        pool: &EmulatedPool,
        order: &[usize],
        g: &Self::Vector,
        x: &Self::Vector,
    ) {
        let count = self.count();
        let order: Vec<u32> = order.iter().map(|&slot| slot as u32).collect();
        let (s, y, g, x) = (pool.s.borrow(), pool.y.borrow(), g.borrow(), x.borrow());
        let mut partials = vec![0.0; ROWS_HISTORY * self.threads];
        for item in 0..self.threads {
            // SAFETY: the vectors, the pools and the partial sums are as the kernel takes them.
            unsafe {
                shared::history_dots(
                    item,
                    count,
                    self.threads,
                    order.len(),
                    order.as_ptr(),
                    s.as_ptr(),
                    y.as_ptr(),
                    g.as_ptr(),
                    x.as_ptr(),
                    partials.as_mut_ptr(),
                )
            }
        }
        self.batch_tail(&partials, ROWS_HISTORY);
    }
    fn update_dots(&self, pool: &EmulatedPool, order: &[usize], slot: usize) {
        let count = self.count();
        let order: Vec<u32> = order.iter().map(|&slot| slot as u32).collect();
        let (s, y) = (pool.s.borrow(), pool.y.borrow());
        let mut partials = vec![0.0; ROWS_UPDATE * self.threads];
        for item in 0..self.threads {
            // SAFETY: the pools and the partial sums are as the kernel takes them.
            unsafe {
                shared::update_dots(
                    item,
                    count,
                    self.threads,
                    order.len(),
                    order.as_ptr(),
                    s.as_ptr(),
                    y.as_ptr(),
                    slot,
                    partials.as_mut_ptr(),
                )
            }
        }
        self.batch_tail(&partials, ROWS_UPDATE);
    }
    fn read_batch(&self, rows: usize) -> Vec<f64> {
        self.batch.borrow()[..rows].to_vec()
    }
    fn combine(
        &self,
        pool: &EmulatedPool,
        order: &[usize],
        coefficients: &[f64; 1 + 2 * HISTORY],
        g: &Self::Vector,
        d: &Self::Vector,
    ) {
        let count = self.count();
        let order: Vec<u32> = order.iter().map(|&slot| slot as u32).collect();
        let (s, y, g, mut d) = (pool.s.borrow(), pool.y.borrow(), g.borrow(), d.borrow_mut());
        for item in 0..count {
            // SAFETY: the vectors and the pools are as the kernel takes them.
            unsafe {
                shared::combine(
                    item,
                    count,
                    order.len(),
                    order.as_ptr(),
                    coefficients.as_ptr(),
                    s.as_ptr(),
                    y.as_ptr(),
                    g.as_ptr(),
                    d.as_mut_ptr(),
                )
            }
        }
    }
}

impl Emulated<'_> {
    /// The passes that take the partial sums of a batch to its dot products.
    fn batch_tail(&self, partials: &[f64], rows: usize) {
        let mut count = self.threads;
        let mut values = partials.to_vec();
        while count > 1 {
            let groups = count.div_ceil(self.group);
            let mut sums = vec![0.0; rows * groups];
            for item in 0..rows * groups {
                // SAFETY: the values are `rows * count` and the sums are `rows * groups`.
                unsafe {
                    shared::sum_columns(
                        item,
                        count,
                        self.group,
                        rows,
                        values.as_ptr(),
                        sums.as_mut_ptr(),
                    )
                }
            }
            values = sums;
            count = groups;
        }
        self.batch.borrow_mut()[..rows].copy_from_slice(&values);
    }
}

#[test]
fn the_compact_form_is_the_two_loop_recursion() {
    // Corrections made of a quadratic, so that each has curvature, on vectors of a dozen values.
    let n = 12;
    let value = |seed: usize, i: usize| ((seed * 31 + i * 17) as f64 * 0.7).sin();
    for m in 1..=HISTORY {
        let s: Vec<Vec<f64>> = (0..m)
            .map(|k| (0..n).map(|i| value(k + 1, i)).collect())
            .collect();
        let y: Vec<Vec<f64>> = (0..m)
            .map(|k| {
                // y = A s with a positive definite A, so that s . y > 0.
                (0..n)
                    .map(|i| {
                        (1.0 + 0.1 * i as f64) * s[k][i]
                            + 0.05 * s[k][(i + 1) % n]
                            + 0.05 * s[k][(i + n - 1) % n]
                    })
                    .collect()
            })
            .collect();
        let g: Vec<f64> = (0..n).map(|i| value(99, i)).collect();
        let dot = |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(a, b)| a * b).sum::<f64>();
        let gamma = dot(&s[m - 1], &y[m - 1]) / dot(&y[m - 1], &y[m - 1]);
        // The two-loop recursion.
        let mut q = g.clone();
        let mut alphas = vec![0.0; m];
        for k in (0..m).rev() {
            alphas[k] = dot(&s[k], &q) / dot(&s[k], &y[k]);
            q.iter_mut()
                .zip(&y[k])
                .for_each(|(q, y)| *q -= alphas[k] * y);
        }
        q.iter_mut().for_each(|q| *q *= gamma);
        for k in 0..m {
            let beta = dot(&y[k], &q) / dot(&s[k], &y[k]);
            q.iter_mut()
                .zip(&s[k])
                .for_each(|(q, s)| *q += (alphas[k] - beta) * s);
        }
        let expected: Vec<f64> = q.iter().map(|q| -q).collect();
        // The compact form.
        let (mut sy, mut yy) = ([[0.0; HISTORY]; HISTORY], [[0.0; HISTORY]; HISTORY]);
        let (mut a, mut b) = ([0.0; HISTORY], [0.0; HISTORY]);
        for i in 0..m {
            a[i] = dot(&s[i], &g);
            b[i] = dot(&y[i], &g);
            for j in 0..m {
                sy[i][j] = dot(&s[i], &y[j]);
                yy[i][j] = dot(&y[i], &y[j]);
            }
        }
        let (coefficients, slope) = compact(m, &sy, &yy, &a, &b, gamma, dot(&g, &g));
        let direction: Vec<f64> = (0..n)
            .map(|i| {
                coefficients[0] * g[i]
                    + (0..m)
                        .map(|k| {
                            coefficients[1 + k] * s[k][i] + coefficients[1 + HISTORY + k] * y[k][i]
                        })
                        .sum::<f64>()
            })
            .collect();
        let scale = expected.iter().fold(0.0_f64, |scale, e| scale.max(e.abs()));
        let error = direction
            .iter()
            .zip(&expected)
            .map(|(d, e)| (d - e).abs())
            .fold(0.0_f64, f64::max);
        assert!(error < 1.0e-10 * scale, "{m} corrections: error {error}");
        assert!(
            (slope - dot(&g, &expected)).abs() < 1.0e-10 * slope.abs(),
            "{m} corrections: slope {slope} against {}",
            dot(&g, &expected)
        );
    }
}

#[test]
fn the_compact_solver_on_the_kernels_follows_the_reference() {
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
    let trace = lbfgs_compact(&backend, &mut x, typical, settings);
    let (difference, same_steps) = compare(&trace, &expected);
    assert!(same_steps, "{:?} against {:?}", trace.steps, expected.steps);
    assert!(difference < 1.0e-8, "the energies differ by {difference}");
    assert_eq!(trace.evaluations, expected.evaluations);
    let error = super::disagreement(&expected_x, &points_of(&backend.download(&x)));
    assert!(error < 1.0e-6, "the points differ by {error}");
}

#[test]
fn the_compact_solver_does_not_depend_on_how_the_dot_products_are_batched() {
    let problem = Problem::new(6, 1.0);
    let layout = Layout::new(&problem);
    let typical = problem.lengths.iter().sum::<f64>() / problem.nodes() as f64;
    let settings = Settings {
        steps: 20,
        ..Default::default()
    };
    let run = |threads: usize, group: usize| {
        let backend = Emulated::with_batch(&layout, threads, group);
        let mut x = backend.vector();
        backend.upload(&x, &planes(&problem.coordinates));
        let trace = lbfgs_compact(&backend, &mut x, typical, settings);
        (trace, backend.download(&x))
    };
    let (expected, expected_x) = run(BATCH_THREADS, BATCH_GROUP);
    // Few threads, many, and groups that do not divide the threads, so that the last group of a
    // pass is short, and a pass of a single group straight away.
    for (threads, group) in [
        (2, 2),
        (7, 3),
        (1000, 7),
        (64, 4),
        (4096, 64),
        (100_000, 33),
        (5, 100),
    ] {
        let (trace, x) = run(threads, group);
        let (difference, same_steps) = compare(&trace, &expected);
        assert!(same_steps, "{threads} threads, groups of {group}");
        assert!(
            difference < 1.0e-9,
            "{threads} threads, groups of {group}: the energies differ by {difference}"
        );
        let error = x
            .iter()
            .zip(&expected_x)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            error < 1.0e-8,
            "{threads} threads, groups of {group}: {error}"
        );
    }
}

#[test]
fn the_automatic_batch_is_what_was_fastest() {
    // The sizes that were timed: 22, 46, 90, 150 and 300 cells on a side, three values a node.
    for (count, threads, group) in [
        (36_501, 16384, 32),
        (311_469, 32768, 32),
        (2_260_713, 131072, 64),
        (10_328_853, 131072, 64),
        (81_812_703, 131072, 64),
    ] {
        let found = automatic_threads(count);
        assert_eq!(
            (found, automatic_group(found)),
            (threads, group),
            "{count} values"
        );
    }
}

// The sweeps.
//
// The fit of the library is not one descent but up to fifty, each of which starts by finding the
// sizes of the nodes and of the elements and the targets of the faces from where the nodes are,
// measures the quality of the elements, and sets the regularization from how it has changed. Each
// goes on for at most a hundred steps, and the fit stops when a descent has nothing to do, when
// the nodes hardly moved, or when the energy has stopped changing.

const RELAXATION: f64 = 0.1;
const EPSILON_FLOOR: f64 = 1.0e-12;
const WINDOW: usize = 3;
const STAGNATION: f64 = 5.0e-4;
const TOLERANCE: f64 = 1.0e-3;

/// The regularization of the next sweep, from the quality of the elements and the least
/// determinant at its start, and the quality at the start of the sweep before.
pub fn schedule(epsilon: f64, quality: f64, previous: f64, worst: f64) -> f64 {
    let sigma = RELAXATION.max(1.0 - quality / previous);
    let mu = (1.0 - sigma) * chi(epsilon, worst);
    let epsilon_2021 = if worst < mu {
        2.0 * (mu * (mu - worst)).sqrt()
    } else {
        EPSILON_FLOOR
    };
    let epsilon_1999 = (1.0e-18 + (0.2 * worst).powi(2)).sqrt();
    epsilon_2021.min(epsilon_1999)
}

/// What a backend does at the start and the end of a sweep.
pub trait SweepBackend: CompactBackend {
    /// Finds the sizes of the nodes and of the elements and the targets of the faces from `x`, and
    /// gives the mean size of the nodes.
    fn start_sweep(&self, x: &Self::Vector) -> f64;
    /// The quality of the elements at `x`, which is their energy without that of the boundary, and
    /// the least determinant of a corner over the cube of the size of its element.
    fn measure(&self, x: &Self::Vector) -> (f64, f64);
    fn set_epsilon(&self, epsilon: f64);
    /// The greatest of how far a node is from where it was, over its size.
    fn shift(&self, start: &Self::Vector, x: &Self::Vector) -> f64;
    /// The seconds so far of finding the targets, where they are found outside the backend.
    #[cfg_attr(not(feature = "geometry"), allow(dead_code))]
    fn target_seconds(&self) -> f64 {
        0.0
    }
}

/// A sweep.
#[derive(Clone, Debug, Default)]
#[cfg_attr(not(feature = "gpu"), allow(dead_code))]
pub struct Sweep {
    pub epsilon: f64,
    pub quality: f64,
    pub worst: f64,
    /// The passes of the descent, those that took a step and those that did not.
    pub passes: usize,
    /// The energy at its end.
    pub value: f64,
    pub shift: f64,
    /// The seconds of finding the sizes and the targets, of measuring, of the descent, and of
    /// the shift.
    pub seconds: [f64; 4],
}

/// What a fit did.
#[derive(Debug, Default)]
pub struct Sweeps {
    pub sweeps: Vec<Sweep>,
    pub stop: &'static str,
}

/// Whether a sweep ends the fit, and why.
fn stops(settled: bool, shift: f64, value: f64, window: &VecDeque<f64>) -> Option<&'static str> {
    let stagnant = window.len() == WINDOW && {
        let high = window.iter().fold(value, |high, &entry| high.max(entry));
        let low = window.iter().fold(value, |low, &entry| low.min(entry));
        high - low <= value.abs() * STAGNATION
    };
    if settled {
        Some("a descent that had nothing to do")
    } else if shift < TOLERANCE {
        Some("the nodes hardly moved")
    } else if stagnant {
        Some("the energy stopped changing")
    } else {
        None
    }
}

/// Fits as the library does, in up to `limit` sweeps, from `epsilon` the first time.
pub fn fit<B: SweepBackend>(
    backend: &B,
    x: &mut B::Vector,
    settings: Settings,
    limit: usize,
    epsilon: f64,
) -> Sweeps {
    let b = backend;
    let start = b.vector();
    let mut epsilon = epsilon;
    let mut previous = f64::INFINITY;
    let mut window = VecDeque::with_capacity(WINDOW);
    let mut result = Sweeps {
        stop: "the sweeps ran out",
        ..Default::default()
    };
    b.set_epsilon(epsilon);
    for sweep in 0..limit {
        let mut seconds = [0.0; 4];
        let started = Instant::now();
        let typical = b.start_sweep(x);
        seconds[0] = started.elapsed().as_secs_f64();
        let started = Instant::now();
        let (quality, worst) = b.measure(x);
        if sweep > 0 {
            epsilon = schedule(epsilon, quality, previous, worst);
            b.set_epsilon(epsilon);
        }
        previous = quality;
        seconds[1] = started.elapsed().as_secs_f64();
        b.scale(1.0, x, &start);
        let started = Instant::now();
        let trace = lbfgs_compact(b, x, typical, settings);
        seconds[2] = started.elapsed().as_secs_f64();
        let value = trace.values.last().copied().unwrap_or(trace.initial);
        let started = Instant::now();
        let shift = b.shift(&start, x);
        seconds[3] = started.elapsed().as_secs_f64();
        let stop = stops(trace.converged && trace.passes == 0, shift, value, &window);
        result.sweeps.push(Sweep {
            epsilon,
            quality,
            worst,
            passes: trace.passes,
            value,
            shift,
            seconds,
        });
        if let Some(reason) = stop {
            result.stop = reason;
            break;
        }
        if window.len() == WINDOW {
            window.pop_front();
        }
        window.push_back(value);
    }
    result
}

/// The same fit with plain loops, on the points, which finds the sizes and the targets of the
/// problem again for each sweep.
pub fn fit_reference(
    problem: &mut Problem,
    start: &[Point],
    settings: Settings,
    limit: usize,
) -> (Sweeps, Vec<Point>) {
    let mut x = start.to_vec();
    let mut epsilon = problem.epsilon;
    let mut previous = f64::INFINITY;
    let mut window = VecDeque::with_capacity(WINDOW);
    let mut result = Sweeps {
        stop: "the sweeps ran out",
        ..Default::default()
    };
    for sweep in 0..limit {
        problem.refresh(&x);
        let typical = problem.lengths.iter().sum::<f64>() / problem.nodes() as f64;
        problem.epsilon = epsilon;
        let (quality, worst) = problem.measure(&x);
        if sweep > 0 {
            epsilon = schedule(epsilon, quality, previous, worst);
            problem.epsilon = epsilon;
        }
        previous = quality;
        let anchor = x.clone();
        let (trace, next) = reference(problem, &x, typical, settings);
        x = next;
        let value = trace.values.last().copied().unwrap_or(trace.initial);
        let shift = x
            .iter()
            .zip(&anchor)
            .zip(&problem.lengths)
            .map(|((x, anchor), length)| {
                let d = minus(*x, *anchor);
                dot(d, d).sqrt() / length
            })
            .fold(0.0, f64::max);
        let stop = stops(trace.converged && trace.passes == 0, shift, value, &window);
        result.sweeps.push(Sweep {
            epsilon,
            quality,
            worst,
            passes: trace.passes,
            value,
            shift,
            seconds: [0.0; 4],
        });
        if let Some(reason) = stop {
            result.stop = reason;
            break;
        }
        if window.len() == WINDOW {
            window.pop_front();
        }
        window.push_back(value);
    }
    (result, x)
}

impl SweepBackend for Emulated<'_> {
    fn start_sweep(&self, x: &Self::Vector) -> f64 {
        let x = x.borrow();
        let mut layout = self.layout.borrow_mut();
        let (nodes, elements, faces) = (layout.nodes, layout.elements, layout.faces);
        {
            let Layout {
                neighbor_starts,
                neighbor_entries,
                lengths,
                ..
            } = &mut *layout;
            for item in 0..nodes {
                // SAFETY: the arrays are those of the layout.
                unsafe {
                    shared::node_lengths(
                        item,
                        nodes,
                        neighbor_starts.as_ptr(),
                        neighbor_entries.as_ptr(),
                        x.as_ptr(),
                        lengths.as_mut_ptr(),
                    )
                }
            }
        }
        {
            let Layout {
                connectivity,
                lengths,
                scales,
                ..
            } = &mut *layout;
            for item in 0..elements {
                // SAFETY: the arrays are those of the layout.
                unsafe {
                    shared::element_scales(
                        item,
                        elements,
                        nodes,
                        connectivity.as_ptr(),
                        lengths.as_ptr(),
                        scales.as_mut_ptr(),
                    )
                }
            }
        }
        if let Some(targets) = &self.targets {
            let (points, normals, distances) = targets(&x);
            layout.points = points;
            layout.normals = normals;
            layout.distances = distances;
        } else {
            let Layout {
                face_nodes,
                points,
                normals,
                distances,
                ..
            } = &mut *layout;
            for item in 0..faces {
                // SAFETY: the arrays are those of the layout.
                unsafe {
                    shared::face_targets(
                        item,
                        faces,
                        nodes,
                        face_nodes.as_ptr(),
                        x.as_ptr(),
                        super::RADIUS,
                        points.as_mut_ptr(),
                        normals.as_mut_ptr(),
                        distances.as_mut_ptr(),
                    )
                }
            }
        }
        sum_by_passes(&layout.lengths, self.threads, self.group) / nodes as f64
    }
    fn measure(&self, x: &Self::Vector) -> (f64, f64) {
        let x = x.borrow();
        let layout = self.layout.borrow();
        let (nodes, elements) = (layout.nodes, layout.elements);
        let mut energies = vec![0.0; elements];
        let mut worst = vec![0.0; elements];
        for item in 0..elements {
            // SAFETY: the arrays are those of the layout.
            unsafe {
                shared::element_energies(
                    item,
                    elements,
                    nodes,
                    layout.connectivity.as_ptr(),
                    x.as_ptr(),
                    layout.scales.as_ptr(),
                    layout.epsilon,
                    energies.as_mut_ptr(),
                );
                shared::element_worst(
                    item,
                    elements,
                    nodes,
                    layout.connectivity.as_ptr(),
                    x.as_ptr(),
                    layout.scales.as_ptr(),
                    worst.as_mut_ptr(),
                )
            }
        }
        (
            sum_by_passes(&energies, self.threads, self.group),
            min_by_passes(&worst, self.threads, self.group),
        )
    }
    fn set_epsilon(&self, epsilon: f64) {
        self.layout.borrow_mut().epsilon = epsilon;
    }
    fn shift(&self, start: &Self::Vector, x: &Self::Vector) -> f64 {
        let (start, x) = (start.borrow(), x.borrow());
        let layout = self.layout.borrow();
        let nodes = layout.nodes;
        let mut values = vec![0.0; nodes];
        for item in 0..nodes {
            // SAFETY: the vectors and the sizes are those of the layout.
            unsafe {
                shared::shift_values(
                    item,
                    nodes,
                    start.as_ptr(),
                    x.as_ptr(),
                    layout.lengths.as_ptr(),
                    values.as_mut_ptr(),
                )
            }
        }
        -min_by_passes(&values, self.threads, self.group)
    }
}

#[test]
fn the_sweeps_on_the_kernels_follow_the_reference() {
    let mut problem = Problem::new(6, 1.0);
    let layout = Layout::new(&problem);
    let start = problem.coordinates.clone();
    let settings = Settings {
        steps: 25,
        ..Default::default()
    };
    let (expected, expected_x) = fit_reference(&mut problem, &start, settings, 4);
    let backend = Emulated::new(&layout);
    let mut x = backend.vector();
    backend.upload(&x, &planes(&start));
    let found = fit(&backend, &mut x, settings, 4, layout.epsilon);
    if std::env::var_os("SHOW").is_some() {
        found.sweeps.iter().for_each(|sweep| eprintln!("{sweep:?}"));
        eprintln!("{}", found.stop);
    }
    assert_eq!(found.sweeps.len(), expected.sweeps.len());
    assert!(found.sweeps.len() >= 2, "{:?}", found.sweeps);
    let close = |a: f64, b: f64, tolerance: f64| (a - b).abs() <= tolerance * b.abs();
    for (index, (a, b)) in found.sweeps.iter().zip(&expected.sweeps).enumerate() {
        assert_eq!(a.passes, b.passes, "sweep {index}");
        assert!(
            close(a.epsilon, b.epsilon, 1.0e-6),
            "sweep {index}: {a:?} against {b:?}"
        );
        assert!(
            close(a.quality, b.quality, 1.0e-8),
            "sweep {index}: {a:?} against {b:?}"
        );
        assert!(
            close(a.worst, b.worst, 1.0e-8),
            "sweep {index}: {a:?} against {b:?}"
        );
        assert!(
            close(a.value, b.value, 1.0e-6),
            "sweep {index}: {a:?} against {b:?}"
        );
        assert!(
            close(a.shift, b.shift, 1.0e-5),
            "sweep {index}: {a:?} against {b:?}"
        );
    }
    assert_eq!(found.stop, expected.stop);
    let error = super::disagreement(&expected_x, &points_of(&backend.download(&x)));
    assert!(error < 1.0e-5, "the points differ by {error}");
}

// The host.
//
// The same fit on the threads of the host, as the other side of a comparison with a device. The
// arithmetic of the kernels for each element, node and face is run on the threads, since each
// writes only what is its own, as the threads of a device do. The work on whole vectors is done in
// contiguous runs, one to a thread, which is what a host does best, and not by the strided threads
// of a device, so its sums are added in another order than the device's.

/// A pointer to what threads read.
#[derive(Clone, Copy)]
struct Reader<T>(*const T);

/// A pointer to what threads write, each in its own places.
#[derive(Clone, Copy)]
struct Writer<T>(*mut T);

// SAFETY: the threads that share these only read what none of them writes, or write to places
// that belong to one of them alone, which is the property that lets the kernels run on a device.
unsafe impl<T> Send for Reader<T> {}
unsafe impl<T> Sync for Reader<T> {}
unsafe impl<T> Send for Writer<T> {}
unsafe impl<T> Sync for Writer<T> {}

impl<T> Reader<T> {
    fn get(self) -> *const T {
        self.0
    }
}

impl<T> Writer<T> {
    fn get(self) -> *mut T {
        self.0
    }
}

/// Calls `work(first, last)` for runs of the items `0..count`, a run to a thread, and gives what
/// the runs return in the order of the runs.
fn collect<T: Send>(
    threads: usize,
    count: usize,
    work: impl Fn(usize, usize) -> T + Sync,
) -> Vec<T> {
    let workers = threads.clamp(1, count.max(1));
    let per = count.div_ceil(workers).max(1);
    let bounds: Vec<(usize, usize)> = (0..workers)
        .map(|worker| (worker * per, ((worker + 1) * per).min(count)))
        .filter(|(first, last)| first < last)
        .collect();
    if bounds.len() <= 1 {
        return bounds
            .iter()
            .map(|&(first, last)| work(first, last))
            .collect();
    }
    scope(|scope| {
        let work = &work;
        bounds
            .iter()
            .map(|&(first, last)| scope.spawn(move || work(first, last)))
            .collect::<Vec<_>>()
            .into_iter()
            .map(|handle| handle.join().unwrap())
            .collect()
    })
}

/// Gives each run of `out`, with the position of its first value, to `work`, a run to a thread.
fn fill(threads: usize, out: &mut [f64], work: impl Fn(usize, &mut [f64]) + Sync) {
    let workers = threads.clamp(1, out.len().max(1));
    let per = out.len().div_ceil(workers).max(1);
    if workers <= 1 {
        work(0, out);
        return;
    }
    scope(|scope| {
        let work = &work;
        for (run, chunk) in out.chunks_mut(per).enumerate() {
            scope.spawn(move || work(run * per, chunk));
        }
    });
}

fn parallel_sum(threads: usize, values: &[f64]) -> f64 {
    collect(threads, values.len(), |first, last| {
        values[first..last].iter().sum::<f64>()
    })
    .into_iter()
    .fold(0.0, |sum, part| sum + part)
}

fn parallel_min(threads: usize, values: &[f64]) -> f64 {
    collect(threads, values.len(), |first, last| {
        values[first..last]
            .iter()
            .fold(f64::INFINITY, |least, &value| least.min(value))
    })
    .into_iter()
    .fold(f64::INFINITY, f64::min)
}

/// The fit on the threads of the host.
pub struct Threaded<'a> {
    layout: RefCell<Layout>,
    threads: usize,
    targets: Option<Targets<'a>>,
    target_seconds: Cell<f64>,
    dots: RefCell<[f64; SLOTS]>,
    alphas: RefCell<[f64; HISTORY + 1]>,
    betas: RefCell<[f64; HISTORY + 1]>,
    batch: RefCell<[f64; BATCH]>,
    /// The energies of the elements and then of the nodes, the contributions of the corners, and
    /// room for a value for each element or node, which are kept from call to call.
    energies: RefCell<Vec<f64>>,
    contributions: RefCell<Vec<f64>>,
    scratch: RefCell<Vec<f64>>,
}

impl<'a> Threaded<'a> {
    /// The fit of a problem on the threads that the system has, or `THREADS` of them.
    pub fn new(layout: Layout) -> Self {
        let threads = std::env::var("THREADS")
            .ok()
            .and_then(|threads| threads.parse().ok())
            .unwrap_or_else(|| std::thread::available_parallelism().map_or(1, |n| n.get()));
        let (nodes, elements) = (layout.nodes, layout.elements);
        Self {
            threads,
            targets: None,
            target_seconds: Cell::new(0.0),
            dots: RefCell::new([0.0; SLOTS]),
            alphas: RefCell::new([0.0; HISTORY + 1]),
            betas: RefCell::new([0.0; HISTORY + 1]),
            batch: RefCell::new([0.0; BATCH]),
            energies: RefCell::new(vec![0.0; elements + nodes]),
            contributions: RefCell::new(vec![0.0; 24 * elements]),
            scratch: RefCell::new(vec![0.0; elements.max(nodes)]),
            layout: RefCell::new(layout),
        }
    }
    pub fn with_threads(mut self, threads: usize) -> Self {
        self.threads = threads.max(1);
        self
    }
    /// With the targets of the faces found by `targets` from where the nodes are, in place of
    /// those of a sphere.
    #[cfg_attr(not(feature = "geometry"), allow(dead_code))]
    pub fn with_targets(mut self, targets: Targets<'a>) -> Self {
        self.targets = Some(targets);
        self
    }
    #[cfg_attr(not(feature = "geometry"), allow(dead_code))]
    pub fn threads(&self) -> usize {
        self.threads
    }
}

impl Backend for Threaded<'_> {
    type Vector = RefCell<Vec<f64>>;
    fn count(&self) -> usize {
        3 * self.layout.borrow().nodes
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
        let layout = self.layout.borrow();
        let x = x.borrow();
        let mut energies = self.energies.borrow_mut();
        let (nodes, elements, faces) = (layout.nodes, layout.elements, layout.faces);
        let out = Writer(energies.as_mut_ptr());
        let coordinates = Reader(x.as_ptr());
        let (connectivity, scales) = (
            Reader(layout.connectivity.as_ptr()),
            Reader(layout.scales.as_ptr()),
        );
        let epsilon = layout.epsilon;
        collect(self.threads, elements, |first, last| {
            for item in first..last {
                // SAFETY: the arrays are those of the layout, and each item writes its own value.
                unsafe {
                    shared::element_energies(
                        item,
                        elements,
                        nodes,
                        connectivity.get(),
                        coordinates.get(),
                        scales.get(),
                        epsilon,
                        out.get(),
                    )
                }
            }
        });
        let (face_starts, face_entries) = (
            Reader(layout.face_starts.as_ptr()),
            Reader(layout.face_entries.as_ptr()),
        );
        let (points, normals, distances, lengths) = (
            Reader(layout.points.as_ptr()),
            Reader(layout.normals.as_ptr()),
            Reader(layout.distances.as_ptr()),
            Reader(layout.lengths.as_ptr()),
        );
        collect(self.threads, nodes, |first, last| {
            for item in first..last {
                // SAFETY: the arrays are those of the layout, and each item writes its own value.
                unsafe {
                    shared::node_energies(
                        item,
                        nodes,
                        elements,
                        face_starts.get(),
                        face_entries.get(),
                        faces,
                        points.get(),
                        normals.get(),
                        distances.get(),
                        coordinates.get(),
                        lengths.get(),
                        out.get(),
                    )
                }
            }
        });
        parallel_sum(self.threads, &energies)
    }
    fn gradient(&self, x: &Self::Vector, gradient: &Self::Vector) {
        let layout = self.layout.borrow();
        let x = x.borrow();
        let mut contributions = self.contributions.borrow_mut();
        let mut gradient = gradient.borrow_mut();
        let (nodes, elements, faces) = (layout.nodes, layout.elements, layout.faces);
        let coordinates = Reader(x.as_ptr());
        let (connectivity, scales) = (
            Reader(layout.connectivity.as_ptr()),
            Reader(layout.scales.as_ptr()),
        );
        let epsilon = layout.epsilon;
        let written = Writer(contributions.as_mut_ptr());
        collect(self.threads, elements, |first, last| {
            for item in first..last {
                // SAFETY: the arrays are those of the layout, and each item writes its own corners.
                unsafe {
                    shared::corner_contributions(
                        item,
                        elements,
                        nodes,
                        connectivity.get(),
                        coordinates.get(),
                        scales.get(),
                        epsilon,
                        written.get(),
                    )
                }
            }
        });
        let read = Reader(contributions.as_ptr());
        let (starts, entries, face_starts, face_entries) = (
            Reader(layout.starts.as_ptr()),
            Reader(layout.entries.as_ptr()),
            Reader(layout.face_starts.as_ptr()),
            Reader(layout.face_entries.as_ptr()),
        );
        let (points, normals, distances, lengths) = (
            Reader(layout.points.as_ptr()),
            Reader(layout.normals.as_ptr()),
            Reader(layout.distances.as_ptr()),
            Reader(layout.lengths.as_ptr()),
        );
        let out = Writer(gradient.as_mut_ptr());
        collect(self.threads, nodes, |first, last| {
            for item in first..last {
                // SAFETY: the arrays are those of the layout, and each item writes its own node.
                unsafe {
                    shared::gradient(
                        item,
                        nodes,
                        elements,
                        starts.get(),
                        entries.get(),
                        read.get(),
                        face_starts.get(),
                        face_entries.get(),
                        faces,
                        points.get(),
                        normals.get(),
                        distances.get(),
                        coordinates.get(),
                        lengths.get(),
                        out.get(),
                    )
                }
            }
        });
    }
    fn axpy(&self, alpha: f64, x: &Self::Vector, y: &Self::Vector, out: &Self::Vector) {
        let (x, y) = (x.borrow(), y.borrow());
        let (x, y): (&[f64], &[f64]) = (&x, &y);
        fill(self.threads, &mut out.borrow_mut(), |first, chunk| {
            for (k, out) in chunk.iter_mut().enumerate() {
                *out = y[first + k] + alpha * x[first + k];
            }
        });
    }
    fn scale(&self, alpha: f64, x: &Self::Vector, out: &Self::Vector) {
        let x = x.borrow();
        let x: &[f64] = &x;
        fill(self.threads, &mut out.borrow_mut(), |first, chunk| {
            for (k, out) in chunk.iter_mut().enumerate() {
                *out = alpha * x[first + k];
            }
        });
    }
    fn scale_in_place(&self, alpha: f64, y: &Self::Vector) {
        fill(self.threads, &mut y.borrow_mut(), |_, chunk| {
            chunk.iter_mut().for_each(|value| *value *= alpha)
        });
    }
    fn dot(&self, a: &Self::Vector, b: &Self::Vector, slot: usize) {
        let (a, b) = (a.borrow(), b.borrow());
        let (a, b): (&[f64], &[f64]) = (&a, &b);
        let total = collect(self.threads, a.len(), |first, last| {
            let mut sum = 0.0;
            for index in first..last {
                sum += a[index] * b[index];
            }
            sum
        })
        .into_iter()
        .fold(0.0, |sum, part| sum + part);
        self.dots.borrow_mut()[slot] = total;
    }
    fn read(&self) -> [f64; SLOTS] {
        *self.dots.borrow()
    }
    fn times(&self, factor: f64, slot: usize, beta: bool, index: usize) {
        let value = factor * self.dots.borrow()[slot];
        if beta {
            self.betas.borrow_mut()[index] = value;
        } else {
            self.alphas.borrow_mut()[index] = value;
        }
    }
    fn subtract_alpha(&self, index: usize, x: &Self::Vector, y: &Self::Vector) {
        let alpha = self.alphas.borrow()[index];
        let x = x.borrow();
        let x: &[f64] = &x;
        fill(self.threads, &mut y.borrow_mut(), |first, chunk| {
            for (k, y) in chunk.iter_mut().enumerate() {
                *y -= alpha * x[first + k];
            }
        });
    }
    fn subtract_pair(&self, index: usize, x: &Self::Vector, y: &Self::Vector) {
        let both = self.alphas.borrow()[index] + self.betas.borrow()[index];
        let x = x.borrow();
        let x: &[f64] = &x;
        fill(self.threads, &mut y.borrow_mut(), |first, chunk| {
            for (k, y) in chunk.iter_mut().enumerate() {
                *y -= both * x[first + k];
            }
        });
    }
}

impl CompactBackend for Threaded<'_> {
    type Pool = EmulatedPool;
    fn pool(&self) -> EmulatedPool {
        EmulatedPool {
            s: RefCell::new(vec![0.0; (HISTORY + 1) * self.count()]),
            y: RefCell::new(vec![0.0; (HISTORY + 1) * self.count()]),
        }
    }
    fn store_step(&self, alpha: f64, x: &Self::Vector, pool: &EmulatedPool, slot: usize) {
        let count = self.count();
        let x = x.borrow();
        let x: &[f64] = &x;
        fill(
            self.threads,
            &mut pool.s.borrow_mut()[slot * count..(slot + 1) * count],
            |first, chunk| {
                for (k, out) in chunk.iter_mut().enumerate() {
                    *out = alpha * x[first + k];
                }
            },
        );
    }
    fn store_change(
        &self,
        new: &Self::Vector,
        old: &Self::Vector,
        pool: &EmulatedPool,
        slot: usize,
    ) {
        let count = self.count();
        let (new, old) = (new.borrow(), old.borrow());
        let (new, old): (&[f64], &[f64]) = (&new, &old);
        fill(
            self.threads,
            &mut pool.y.borrow_mut()[slot * count..(slot + 1) * count],
            |first, chunk| {
                for (k, out) in chunk.iter_mut().enumerate() {
                    *out = new[first + k] - old[first + k];
                }
            },
        );
    }
    fn history_dots(
        &self,
        pool: &EmulatedPool,
        order: &[usize],
        g: &Self::Vector,
        x: &Self::Vector,
    ) {
        let count = self.count();
        let (s, y, g, x) = (pool.s.borrow(), pool.y.borrow(), g.borrow(), x.borrow());
        let (s, y, g, x): (&[f64], &[f64], &[f64], &[f64]) = (&s, &y, &g, &x);
        let positions: Vec<usize> = order.iter().map(|&slot| slot * count).collect();
        let active = positions.len();
        let rows = collect(self.threads, count, |first, last| {
            let mut row = [0.0; ROWS_HISTORY];
            for i in first..last {
                let (gi, xi) = (g[i], x[i]);
                row[0] += gi * gi;
                row[1] += xi * xi;
                for j in 0..active {
                    row[2 + j] += s[positions[j] + i] * gi;
                    row[2 + HISTORY + j] += y[positions[j] + i] * gi;
                }
            }
            row
        });
        let mut batch = self.batch.borrow_mut();
        batch.fill(0.0);
        for row in rows {
            for (total, part) in batch.iter_mut().zip(row) {
                *total += part;
            }
        }
    }
    fn update_dots(&self, pool: &EmulatedPool, order: &[usize], slot: usize) {
        let count = self.count();
        let (s, y) = (pool.s.borrow(), pool.y.borrow());
        let (s, y): (&[f64], &[f64]) = (&s, &y);
        let positions: Vec<usize> = order.iter().map(|&slot| slot * count).collect();
        let active = positions.len();
        let new = slot * count;
        let rows = collect(self.threads, count, |first, last| {
            let mut row = [0.0; ROWS_UPDATE];
            for i in first..last {
                let (sn, yn) = (s[new + i], y[new + i]);
                row[0] += sn * yn;
                row[1] += sn * sn;
                row[2] += yn * yn;
                for j in 0..active {
                    let (so, yo) = (s[positions[j] + i], y[positions[j] + i]);
                    row[3 + j] += so * yn;
                    row[3 + HISTORY + j] += sn * yo;
                    row[3 + 2 * HISTORY + j] += yo * yn;
                }
            }
            row
        });
        let mut batch = self.batch.borrow_mut();
        batch.fill(0.0);
        for row in rows {
            for (total, part) in batch.iter_mut().zip(row) {
                *total += part;
            }
        }
    }
    fn read_batch(&self, rows: usize) -> Vec<f64> {
        self.batch.borrow()[..rows].to_vec()
    }
    fn combine(
        &self,
        pool: &EmulatedPool,
        order: &[usize],
        coefficients: &[f64; 1 + 2 * HISTORY],
        g: &Self::Vector,
        d: &Self::Vector,
    ) {
        let count = self.count();
        let (s, y, g) = (pool.s.borrow(), pool.y.borrow(), g.borrow());
        let (s, y, g): (&[f64], &[f64], &[f64]) = (&s, &y, &g);
        let positions: Vec<usize> = order.iter().map(|&slot| slot * count).collect();
        fill(self.threads, &mut d.borrow_mut(), |first, chunk| {
            for (k, out) in chunk.iter_mut().enumerate() {
                let i = first + k;
                let mut value = coefficients[0] * g[i];
                for (j, &position) in positions.iter().enumerate() {
                    value += coefficients[1 + j] * s[position + i]
                        + coefficients[1 + HISTORY + j] * y[position + i];
                }
                *out = value;
            }
        });
    }
}

impl SweepBackend for Threaded<'_> {
    fn start_sweep(&self, x: &Self::Vector) -> f64 {
        let x = x.borrow();
        let mut layout = self.layout.borrow_mut();
        let (nodes, elements, faces) = (layout.nodes, layout.elements, layout.faces);
        let coordinates = Reader(x.as_ptr());
        {
            let Layout {
                neighbor_starts,
                neighbor_entries,
                lengths,
                ..
            } = &mut *layout;
            let (starts, entries) = (
                Reader(neighbor_starts.as_ptr()),
                Reader(neighbor_entries.as_ptr()),
            );
            let out = Writer(lengths.as_mut_ptr());
            collect(self.threads, nodes, |first, last| {
                for item in first..last {
                    // SAFETY: the arrays are those of the layout, and each item writes its node.
                    unsafe {
                        shared::node_lengths(
                            item,
                            nodes,
                            starts.get(),
                            entries.get(),
                            coordinates.get(),
                            out.get(),
                        )
                    }
                }
            });
        }
        {
            let Layout {
                connectivity,
                lengths,
                scales,
                ..
            } = &mut *layout;
            let (connectivity, lengths) = (Reader(connectivity.as_ptr()), Reader(lengths.as_ptr()));
            let out = Writer(scales.as_mut_ptr());
            collect(self.threads, elements, |first, last| {
                for item in first..last {
                    // SAFETY: the arrays are those of the layout, and each item writes its own.
                    unsafe {
                        shared::element_scales(
                            item,
                            elements,
                            nodes,
                            connectivity.get(),
                            lengths.get(),
                            out.get(),
                        )
                    }
                }
            });
        }
        if let Some(targets) = &self.targets {
            let started = Instant::now();
            let (points, normals, distances) = targets(&x);
            layout.points = points;
            layout.normals = normals;
            layout.distances = distances;
            self.target_seconds
                .set(self.target_seconds.get() + started.elapsed().as_secs_f64());
        } else {
            let Layout {
                face_nodes,
                points,
                normals,
                distances,
                ..
            } = &mut *layout;
            let face_nodes = Reader(face_nodes.as_ptr());
            let (points, normals, distances) = (
                Writer(points.as_mut_ptr()),
                Writer(normals.as_mut_ptr()),
                Writer(distances.as_mut_ptr()),
            );
            collect(self.threads, faces, |first, last| {
                for item in first..last {
                    // SAFETY: the arrays are those of the layout, and each item writes its face.
                    unsafe {
                        shared::face_targets(
                            item,
                            faces,
                            nodes,
                            face_nodes.get(),
                            coordinates.get(),
                            super::RADIUS,
                            points.get(),
                            normals.get(),
                            distances.get(),
                        )
                    }
                }
            });
        }
        parallel_sum(self.threads, &layout.lengths) / nodes as f64
    }
    fn measure(&self, x: &Self::Vector) -> (f64, f64) {
        let x = x.borrow();
        let layout = self.layout.borrow();
        let (nodes, elements) = (layout.nodes, layout.elements);
        let mut energies = self.energies.borrow_mut();
        let mut worst = self.scratch.borrow_mut();
        let (coordinates, connectivity, scales) = (
            Reader(x.as_ptr()),
            Reader(layout.connectivity.as_ptr()),
            Reader(layout.scales.as_ptr()),
        );
        let epsilon = layout.epsilon;
        let (out_energy, out_worst) = (Writer(energies.as_mut_ptr()), Writer(worst.as_mut_ptr()));
        collect(self.threads, elements, |first, last| {
            for item in first..last {
                // SAFETY: the arrays are those of the layout, and each item writes its own values.
                unsafe {
                    shared::element_energies(
                        item,
                        elements,
                        nodes,
                        connectivity.get(),
                        coordinates.get(),
                        scales.get(),
                        epsilon,
                        out_energy.get(),
                    );
                    shared::element_worst(
                        item,
                        elements,
                        nodes,
                        connectivity.get(),
                        coordinates.get(),
                        scales.get(),
                        out_worst.get(),
                    )
                }
            }
        });
        (
            parallel_sum(self.threads, &energies[..elements]),
            parallel_min(self.threads, &worst[..elements]),
        )
    }
    fn set_epsilon(&self, epsilon: f64) {
        self.layout.borrow_mut().epsilon = epsilon;
    }
    fn shift(&self, start: &Self::Vector, x: &Self::Vector) -> f64 {
        let (start, x) = (start.borrow(), x.borrow());
        let layout = self.layout.borrow();
        let nodes = layout.nodes;
        let mut values = self.scratch.borrow_mut();
        let (start, x, lengths) = (
            Reader(start.as_ptr()),
            Reader(x.as_ptr()),
            Reader(layout.lengths.as_ptr()),
        );
        let out = Writer(values.as_mut_ptr());
        collect(self.threads, nodes, |first, last| {
            for item in first..last {
                // SAFETY: the vectors and the sizes are those of the layout, each item writes its own.
                unsafe {
                    shared::shift_values(
                        item,
                        nodes,
                        start.get(),
                        x.get(),
                        lengths.get(),
                        out.get(),
                    )
                }
            }
        });
        -parallel_min(self.threads, &values[..nodes])
    }
    fn target_seconds(&self) -> f64 {
        self.target_seconds.get()
    }
}

#[test]
fn the_sweeps_on_the_threads_follow_the_reference() {
    let mut problem = Problem::new(6, 1.0);
    let layout = Layout::new(&problem);
    let start = problem.coordinates.clone();
    let settings = Settings {
        steps: 15,
        ..Default::default()
    };
    let (expected, expected_x) = fit_reference(&mut problem, &start, settings, 3);
    for threads in [1, 3, 8] {
        let backend = Threaded::new(layout.clone()).with_threads(threads);
        let mut x = backend.vector();
        backend.upload(&x, &planes(&start));
        let found = fit(&backend, &mut x, settings, 3, layout.epsilon);
        assert_eq!(
            found.sweeps.len(),
            expected.sweeps.len(),
            "{threads} threads"
        );
        let close = |a: f64, b: f64, tolerance: f64| (a - b).abs() <= 1.0e-6 + tolerance * b.abs();
        for (index, (a, b)) in found.sweeps.iter().zip(&expected.sweeps).enumerate() {
            let at = format!("{threads} threads, sweep {index}: {a:?} against {b:?}");
            assert_eq!(a.passes, b.passes, "{at}");
            assert!(close(a.epsilon, b.epsilon, 1.0e-4), "{at}");
            assert!(close(a.quality, b.quality, 1.0e-6), "{at}");
            assert!(close(a.worst, b.worst, 1.0e-4), "{at}");
            assert!(close(a.value, b.value, 1.0e-5), "{at}");
            assert!(close(a.shift, b.shift, 1.0e-4), "{at}");
        }
        let error = super::disagreement(&expected_x, &points_of(&backend.download(&x)));
        assert!(
            error < 1.0e-4,
            "{threads} threads: the points differ by {error}"
        );
    }
}
