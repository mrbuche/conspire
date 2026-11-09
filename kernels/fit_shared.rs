//! The arithmetic of the kernels of the fit benchmark, for one thread at a time.
//!
//! These functions are what the kernels of `fit.rs` run for each thread on a device. The benchmark
//! also runs them on the host, one thread after another, to check their indexing and their sums
//! without a device. The module that includes this file has the function `sqrt`.
//!
//! The data are on planes, so that threads next to each other read values next to each other:
//!
//! - the coordinates of `nodes` nodes are `coordinates[d * nodes + node]` for the component `d`;
//! - the node `c` of the element `item` of `elements` elements is
//!   `connectivity[c * elements + item]`;
//! - the contribution of that corner to the gradient is `contributions[d * 8 * elements + c *
//!   elements + item]`, so that the position of a corner among the corners is `c * elements +
//!   item`, and the entries of a node are those positions;
//! - the `faces` faces have `points[d * faces + face]` and `normals[d * faces + face]`;
//! - the gradient is `gradient[d * nodes + node]`.

#![allow(clippy::needless_range_loop, clippy::too_many_arguments)]

use super::sqrt;
use core::{
    ops::{Index, IndexMut, Range},
    slice::{from_raw_parts, from_raw_parts_mut},
};

/// Values that are checked against their number in a debug build, as the tests of the host are,
/// and are not in a release build, as the kernels are, where a check is a branch in every thread.
struct View<'a, T>(&'a [T]);

/// The same to write to.
struct ViewMut<'a, T>(&'a mut [T]);

/// # Safety
///
/// `pointer` must be `length` values.
unsafe fn view<'a, T>(pointer: *const T, length: usize) -> View<'a, T> {
    // SAFETY: as the caller says.
    View(unsafe { from_raw_parts(pointer, length) })
}

/// # Safety
///
/// `pointer` must be `length` values, which nothing else is using.
unsafe fn view_mut<'a, T>(pointer: *mut T, length: usize) -> ViewMut<'a, T> {
    // SAFETY: as the caller says.
    ViewMut(unsafe { from_raw_parts_mut(pointer, length) })
}

impl<T> Index<usize> for View<'_, T> {
    type Output = T;
    #[inline(always)]
    fn index(&self, index: usize) -> &T {
        debug_assert!(index < self.0.len(), "{index} of {}", self.0.len());
        // SAFETY: the index is among the values, as the kernel's caller says.
        unsafe { self.0.get_unchecked(index) }
    }
}

impl<T> Index<Range<usize>> for View<'_, T> {
    type Output = [T];
    #[inline(always)]
    fn index(&self, range: Range<usize>) -> &[T] {
        debug_assert!(range.start <= range.end && range.end <= self.0.len());
        // SAFETY: the range is among the values, as the kernel's caller says.
        unsafe { self.0.get_unchecked(range) }
    }
}

impl<T> Index<usize> for ViewMut<'_, T> {
    type Output = T;
    #[inline(always)]
    fn index(&self, index: usize) -> &T {
        debug_assert!(index < self.0.len(), "{index} of {}", self.0.len());
        // SAFETY: the index is among the values, as the kernel's caller says.
        unsafe { self.0.get_unchecked(index) }
    }
}

impl<T> IndexMut<usize> for ViewMut<'_, T> {
    #[inline(always)]
    fn index_mut(&mut self, index: usize) -> &mut T {
        debug_assert!(index < self.0.len(), "{index} of {}", self.0.len());
        // SAFETY: the index is among the values, as the kernel's caller says.
        unsafe { self.0.get_unchecked_mut(index) }
    }
}

type Point = [f64; 3];

/// How hard the boundary is held, as in the fit of the library.
pub const BALANCE: f64 = 2.5e3;
/// The least that the squared distance of a face over the squared size of a node counts for.
pub const WEIGHT_FLOOR: f64 = 0.3;
fn minus(a: Point, b: Point) -> Point {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn dot(a: Point, b: Point) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross(a: Point, b: Point) -> Point {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn chi(epsilon: f64, determinant: f64) -> f64 {
    0.5 * (determinant + sqrt(epsilon * epsilon + determinant * determinant))
}

fn weight(distance: f64, length: f64) -> f64 {
    1.0 / (distance / (length * length)).max(WEIGHT_FLOOR)
}

/// The eight nodes of an element.
///
/// # Safety
///
/// `connectivity` must be `8 * elements` values and `coordinates` must be `3 * nodes`, with the
/// element and its nodes among them.
unsafe fn points(
    item: usize,
    elements: usize,
    nodes: usize,
    connectivity: *const u32,
    coordinates: *const f64,
) -> [Point; 8] {
    // SAFETY: as the caller says.
    let (connectivity, coordinates) = unsafe {
        (
            view(connectivity, 8 * elements),
            view(coordinates, 3 * nodes),
        )
    };
    let mut points = [[0.0; 3]; 8];
    for corner in 0..8 {
        let node = connectivity[corner * elements + item] as usize;
        for d in 0..3 {
            points[corner][d] = coordinates[d * nodes + node];
        }
    }
    points
}

/// The three edges that leave a corner, their squared lengths added, and their scalar triple
/// product.
fn corner(points: &[Point; 8], corner: usize, adjacent: &[usize; 3]) -> ([Point; 3], f64, f64) {
    let origin = points[corner];
    let e = [
        minus(points[adjacent[0]], origin),
        minus(points[adjacent[1]], origin),
        minus(points[adjacent[2]], origin),
    ];
    let trace = dot(e[0], e[0]) + dot(e[1], e[1]) + dot(e[2], e[2]);
    (e, trace, dot(e[0], cross(e[1], e[2])))
}

/// The energy of the corner `C`, with the nodes `A0`, `A1` and `A2` next to it.
#[inline(always)]
fn corner_energy<const C: usize, const A0: usize, const A1: usize, const A2: usize>(
    points: &[Point; 8],
    epsilon: f64,
) -> f64 {
    let (_, trace, determinant) = corner(points, C, &[A0, A1, A2]);
    trace * sqrt(trace) / chi(epsilon, determinant)
}

/// What the corner `C`, with the nodes `A0`, `A1` and `A2` next to it, adds to how the energy of its
/// element changes with each node.
#[inline(always)]
fn corner_term<const C: usize, const A0: usize, const A1: usize, const A2: usize>(
    points: &[Point; 8],
    local: &mut [Point; 8],
    epsilon: f64,
) {
    let adjacent = [A0, A1, A2];
    let (e, trace, determinant) = corner(points, C, &adjacent);
    let denominator = chi(epsilon, determinant);
    let root = sqrt(trace);
    let alpha = 3.0 * root / denominator;
    let beta = trace
        * root
        * 0.5
        * (1.0 + determinant / sqrt(epsilon * epsilon + determinant * determinant))
        / (denominator * denominator);
    let crosses = [cross(e[1], e[2]), cross(e[2], e[0]), cross(e[0], e[1])];
    for i in 0..3 {
        for d in 0..3 {
            local[C][d] += crosses[i][d] * beta - e[i][d] * alpha;
            local[adjacent[i]][d] += e[i][d] * alpha - crosses[i][d] * beta;
        }
    }
}

/// The first pass of the gradient: the thread `item` writes how the energy of its element changes
/// with each of its corners, times the size of the element.
///
/// # Safety
///
/// The arrays must be as the layout says, and `contributions` must be `24 * elements` values.
pub unsafe fn corner_contributions(
    item: usize,
    elements: usize,
    nodes: usize,
    connectivity: *const u32,
    coordinates: *const f64,
    scales: *const f64,
    epsilon: f64,
    contributions: *mut f64,
) {
    if item >= elements {
        return;
    }
    // SAFETY: as the caller says.
    let (points, scales, mut contributions) = unsafe {
        (
            points(item, elements, nodes, connectivity, coordinates),
            view(scales, elements),
            view_mut(contributions, 24 * elements),
        )
    };
    let scale = scales[item];
    let epsilon = scale * scale * scale * epsilon;
    let mut local = [[0.0; 3]; 8];
    // The corners of `CORNERS`, in its order.
    corner_term::<0, 1, 3, 4>(&points, &mut local, epsilon);
    corner_term::<1, 2, 0, 5>(&points, &mut local, epsilon);
    corner_term::<2, 3, 1, 6>(&points, &mut local, epsilon);
    corner_term::<3, 0, 2, 7>(&points, &mut local, epsilon);
    corner_term::<4, 7, 5, 0>(&points, &mut local, epsilon);
    corner_term::<5, 4, 6, 1>(&points, &mut local, epsilon);
    corner_term::<6, 5, 7, 2>(&points, &mut local, epsilon);
    corner_term::<7, 6, 4, 3>(&points, &mut local, epsilon);
    for c in 0..8 {
        for d in 0..3 {
            contributions[(d * 8 + c) * elements + item] = local[c][d] * scale;
        }
    }
}

/// The energy of the element of the thread `item`, times its size, into `energies[item]`.
///
/// # Safety
///
/// The arrays must be as the layout says, and `energies` must have a value for the element.
pub unsafe fn element_energies(
    item: usize,
    elements: usize,
    nodes: usize,
    connectivity: *const u32,
    coordinates: *const f64,
    scales: *const f64,
    epsilon: f64,
    energies: *mut f64,
) {
    if item >= elements {
        return;
    }
    // SAFETY: as the caller says.
    let (points, scales, mut energies) = unsafe {
        (
            points(item, elements, nodes, connectivity, coordinates),
            view(scales, elements),
            view_mut(energies, elements),
        )
    };
    let scale = scales[item];
    let epsilon = scale * scale * scale * epsilon;
    let mut sum = 0.0;
    // The corners of a hexahedron, each with the three nodes that it is joined to.
    sum += corner_energy::<0, 1, 3, 4>(&points, epsilon);
    sum += corner_energy::<1, 2, 0, 5>(&points, epsilon);
    sum += corner_energy::<2, 3, 1, 6>(&points, epsilon);
    sum += corner_energy::<3, 0, 2, 7>(&points, epsilon);
    sum += corner_energy::<4, 7, 5, 0>(&points, epsilon);
    sum += corner_energy::<5, 4, 6, 1>(&points, epsilon);
    sum += corner_energy::<6, 5, 7, 2>(&points, epsilon);
    sum += corner_energy::<7, 6, 4, 3>(&points, epsilon);
    energies[item] = scale * sum;
}

/// How far a node is from the plane of a face, along its normal.
fn deviation(
    node: usize,
    nodes: usize,
    face: usize,
    faces: usize,
    coordinates: &View<f64>,
    points: &View<f64>,
    normals: &View<f64>,
) -> f64 {
    (coordinates[node] - points[face]) * normals[face]
        + (coordinates[nodes + node] - points[faces + face]) * normals[faces + face]
        + (coordinates[2 * nodes + node] - points[2 * faces + face]) * normals[2 * faces + face]
}

/// The energy of the thread `item`, a node, with the faces it is on, into `energies[offset +
/// item]`.
///
/// # Safety
///
/// The arrays must be as the layout says, `face_starts` must be `nodes + 1` values, and `energies`
/// must have a value at `offset + item`.
pub unsafe fn node_energies(
    item: usize,
    nodes: usize,
    offset: usize,
    face_starts: *const u32,
    face_entries: *const u32,
    faces: usize,
    points: *const f64,
    normals: *const f64,
    distances: *const f64,
    coordinates: *const f64,
    lengths: *const f64,
    energies: *mut f64,
) {
    if item >= nodes {
        return;
    }
    // SAFETY: as the caller says.
    let (face_starts, points, normals, distances, coordinates, lengths) = unsafe {
        (
            view(face_starts, nodes + 1),
            view(points, 3 * faces),
            view(normals, 3 * faces),
            view(distances, faces),
            view(coordinates, 3 * nodes),
            view(lengths, nodes),
        )
    };
    let (first, last) = (face_starts[item] as usize, face_starts[item + 1] as usize);
    // SAFETY: the entries are as many as the last of the starts says.
    let face_entries = unsafe { view(face_entries, last) };
    let length = lengths[item];
    let mut sum = 0.0;
    for &face in &face_entries[first..last] {
        let face = face as usize;
        let deviation = deviation(item, nodes, face, faces, &coordinates, &points, &normals);
        sum += weight(distances[face], length) * deviation * deviation;
    }
    // SAFETY: the energy of this node has a place.
    unsafe { *energies.add(offset + item) = BALANCE / length * sum };
}

/// The second pass of the gradient: the thread `item`, a node, adds up the contributions of the
/// corners that sit on it in the order of the elements, then the force of the boundary on it.
///
/// # Safety
///
/// The arrays must be as the layout says, `starts` must be `nodes + 1` values, and `gradient` must
/// be `3 * nodes`.
pub unsafe fn gradient(
    item: usize,
    nodes: usize,
    elements: usize,
    starts: *const u32,
    entries: *const u32,
    contributions: *const f64,
    face_starts: *const u32,
    face_entries: *const u32,
    faces: usize,
    points: *const f64,
    normals: *const f64,
    distances: *const f64,
    coordinates: *const f64,
    lengths: *const f64,
    gradient: *mut f64,
) {
    if item >= nodes {
        return;
    }
    // SAFETY: as the caller says.
    let (starts, face_starts, points, normals, distances, coordinates, lengths) = unsafe {
        (
            view(starts, nodes + 1),
            view(face_starts, nodes + 1),
            view(points, 3 * faces),
            view(normals, 3 * faces),
            view(distances, faces),
            view(coordinates, 3 * nodes),
            view(lengths, nodes),
        )
    };
    let plane = 8 * elements;
    let (first, last) = (starts[item] as usize, starts[item + 1] as usize);
    let (first_face, last_face) = (face_starts[item] as usize, face_starts[item + 1] as usize);
    // SAFETY: the entries are as many as the last of the starts say.
    let (entries, face_entries, contributions) = unsafe {
        (
            view(entries, last),
            view(face_entries, last_face),
            view(contributions, 3 * plane),
        )
    };
    let mut sum = [0.0; 3];
    for &entry in &entries[first..last] {
        for d in 0..3 {
            sum[d] += contributions[d * plane + entry as usize];
        }
    }
    let length = lengths[item];
    for &face in &face_entries[first_face..last_face] {
        let face = face as usize;
        let deviation = deviation(item, nodes, face, faces, &coordinates, &points, &normals);
        let factor = 2.0 * BALANCE / length * weight(distances[face], length) * deviation;
        for d in 0..3 {
            sum[d] += normals[d * faces + face] * factor;
        }
    }
    // SAFETY: the gradient has three values for each node.
    let mut gradient = unsafe { view_mut(gradient, 3 * nodes) };
    for d in 0..3 {
        gradient[d * nodes + item] = sum[d];
    }
}

/// The first pass of a sum of `count` values: the thread `item` of `threads` adds up the values
/// that are `threads` apart from the one at `item`, in order, into `partials[item]`.
///
/// # Safety
///
/// `values` must be `count` values and `partials` must be `threads`.
pub unsafe fn sum_strided(
    item: usize,
    count: usize,
    threads: usize,
    values: *const f64,
    partials: *mut f64,
) {
    if item >= threads {
        return;
    }
    // SAFETY: as the caller says.
    let (values, mut partials) = unsafe { (view(values, count), view_mut(partials, threads)) };
    let mut sum = 0.0;
    let mut index = item;
    while index < count {
        sum += values[index];
        index += threads;
    }
    partials[item] = sum;
}

/// A pass of a sum of `count` values after the first: the thread `item` adds up the `group` values
/// from `item * group`, in order, into `sums[item]`.
///
/// # Safety
///
/// `values` must be `count` values and `sums` must be `count.div_ceil(group)`.
pub unsafe fn sum_groups(
    item: usize,
    count: usize,
    group: usize,
    values: *const f64,
    sums: *mut f64,
) {
    let groups = count.div_ceil(group);
    if item >= groups {
        return;
    }
    // SAFETY: as the caller says.
    let (values, mut sums) = unsafe { (view(values, count), view_mut(sums, groups)) };
    let mut sum = 0.0;
    for &value in &values[item * group..((item + 1) * group).min(count)] {
        sum += value;
    }
    sums[item] = sum;
}

/// The first pass of the dot product of two vectors of `count` values: the thread `item` of
/// `threads` adds up the products of the values that are `threads` apart from the one at `item`,
/// in order, into `partials[item]`. The passes after it are those of [`sum_groups`] and
/// [`sum_into`].
///
/// # Safety
///
/// `a` and `b` must be `count` values and `partials` must be `threads`.
pub unsafe fn dot_strided(
    item: usize,
    count: usize,
    threads: usize,
    a: *const f64,
    b: *const f64,
    partials: *mut f64,
) {
    if item >= threads {
        return;
    }
    let mut sum = 0.0;
    let mut index = item;
    while index < count {
        // SAFETY: the index is among the `count` values.
        sum += unsafe { *a.add(index) * *b.add(index) };
        index += threads;
    }
    // SAFETY: there is a partial sum for each thread.
    unsafe { *partials.add(item) = sum };
}

/// The last pass of a sum: the first thread adds up `count` values, in order, into `out[slot]`.
///
/// # Safety
///
/// `values` must be `count` values and `out` must have a value at `slot`.
pub unsafe fn sum_into(item: usize, count: usize, values: *const f64, out: *mut f64, slot: usize) {
    if item != 0 {
        return;
    }
    let mut sum = 0.0;
    for index in 0..count {
        // SAFETY: the index is among the `count` values.
        sum += unsafe { *values.add(index) };
    }
    // SAFETY: the slot is one of `out`.
    unsafe { *out.add(slot) = sum };
}

/// The first thread sets `out[index]` to `factor` times `dots[dot]`, a number that is on the
/// device and is not read back.
///
/// # Safety
///
/// `dots` must have a value at `dot` and `out` must have a value at `index`.
pub unsafe fn times(
    item: usize,
    factor: f64,
    dots: *const f64,
    dot: usize,
    out: *mut f64,
    index: usize,
) {
    if item != 0 {
        return;
    }
    // SAFETY: as the caller says.
    unsafe { *out.add(index) = factor * *dots.add(dot) };
}

/// `out[item] = y[item] + alpha * x[item]`.
///
/// # Safety
///
/// `x`, `y` and `out` must be `count` values.
pub unsafe fn axpy(
    item: usize,
    count: usize,
    alpha: f64,
    x: *const f64,
    y: *const f64,
    out: *mut f64,
) {
    if item >= count {
        return;
    }
    // SAFETY: the item is among the `count` values.
    unsafe { *out.add(item) = *y.add(item) + alpha * *x.add(item) };
}

/// `out[item] = alpha * x[item]`.
///
/// # Safety
///
/// `x` and `out` must be `count` values.
pub unsafe fn scale(item: usize, count: usize, alpha: f64, x: *const f64, out: *mut f64) {
    if item >= count {
        return;
    }
    // SAFETY: the item is among the `count` values.
    unsafe { *out.add(item) = alpha * *x.add(item) };
}

/// `y[item] *= alpha`.
///
/// # Safety
///
/// `y` must be `count` values.
pub unsafe fn scale_in_place(item: usize, count: usize, alpha: f64, y: *mut f64) {
    if item >= count {
        return;
    }
    // SAFETY: the item is among the `count` values.
    unsafe { *y.add(item) *= alpha };
}

/// `y[item] -= scalars[index] * x[item]`, where the scalar is on the device.
///
/// # Safety
///
/// `x` and `y` must be `count` values and `scalars` must have a value at `index`.
pub unsafe fn subtract_alpha(
    item: usize,
    count: usize,
    scalars: *const f64,
    index: usize,
    x: *const f64,
    y: *mut f64,
) {
    if item >= count {
        return;
    }
    // SAFETY: the item is among the `count` values, and the index is one of the scalars.
    unsafe { *y.add(item) -= *scalars.add(index) * *x.add(item) };
}

/// `y[item] -= (alphas[index] + betas[index]) * x[item]`, where the scalars are on the device.
///
/// # Safety
///
/// `x` and `y` must be `count` values and `alphas` and `betas` must have a value at `index`.
pub unsafe fn subtract_pair(
    item: usize,
    count: usize,
    alphas: *const f64,
    betas: *const f64,
    index: usize,
    x: *const f64,
    y: *mut f64,
) {
    if item >= count {
        return;
    }
    // SAFETY: the item is among the `count` values, and the index is one of the scalars.
    unsafe { *y.add(item) -= (*alphas.add(index) + *betas.add(index)) * *x.add(item) };
}

// The kernels of the compact form of the limited-memory BFGS.
//
// The corrections are kept in two pools, one of the steps `s` and one of the changes of the
// gradient `y`, each `CORRECTIONS + 1` vectors of `count` values one after another, so that a
// correction in the slot `slot` is at `slot * count`. The corrections that are in use, oldest
// first, are the slots in `order`.

/// The most corrections that the kernels of the compact form take.
pub const CORRECTIONS: usize = 8;
/// The dot products of the first pass of a step: the gradient with itself, the iterate with
/// itself, and the gradient with the steps and with the changes of the gradient.
#[allow(dead_code)]
pub const ROWS_HISTORY: usize = 2 + 2 * CORRECTIONS;
/// The dot products of the update of the history: the new step with the new change, the new step
/// with itself, the new change with itself, and the new correction with the old ones.
#[allow(dead_code)]
pub const ROWS_UPDATE: usize = 3 + 3 * CORRECTIONS;

/// The positions in the pools of the corrections in use.
///
/// # Safety
///
/// `order` must be `active` values, which are at most `CORRECTIONS`.
unsafe fn positions(active: usize, order: *const u32, count: usize) -> [usize; CORRECTIONS] {
    let mut positions = [0; CORRECTIONS];
    for j in 0..CORRECTIONS {
        if j < active {
            // SAFETY: the correction is one of the `active`.
            positions[j] = unsafe { *order.add(j) } as usize * count;
        }
    }
    positions
}

/// The first pass of the dot products against the history: the thread `item` of `threads` adds
/// up, over the values that are `threads` apart from the one at `item`, the products that make
/// each of the `ROWS_HISTORY` dot products, in order, into `partials[row * threads + item]`. The
/// rows are `g . g`, `x . x`, the dot products of each step in use with `g`, and then those of
/// each change of the gradient in use. The rows of the corrections that are not in use are zero.
///
/// # Safety
///
/// `g` and `x` must be `count` values, the pools must be `CORRECTIONS + 1` vectors of them, the
/// slots in `order` must be among those, and `partials` must be `ROWS_HISTORY * threads`.
pub unsafe fn history_dots(
    item: usize,
    count: usize,
    threads: usize,
    active: usize,
    order: *const u32,
    pool_s: *const f64,
    pool_y: *const f64,
    g: *const f64,
    x: *const f64,
    partials: *mut f64,
) {
    if item >= threads {
        return;
    }
    // SAFETY: as the caller says.
    let positions = unsafe { positions(active, order, count) };
    let (mut gg, mut xx) = (0.0, 0.0);
    let (mut sg, mut yg) = ([0.0; CORRECTIONS], [0.0; CORRECTIONS]);
    let mut index = item;
    while index < count {
        // SAFETY: the index is among the `count` values.
        let (gi, xi) = unsafe { (*g.add(index), *x.add(index)) };
        gg += gi * gi;
        xx += xi * xi;
        for j in 0..CORRECTIONS {
            if j < active {
                // SAFETY: the correction is in the pools, and the index is among its values.
                let (s, y) = unsafe {
                    (
                        *pool_s.add(positions[j] + index),
                        *pool_y.add(positions[j] + index),
                    )
                };
                sg[j] += s * gi;
                yg[j] += y * gi;
            }
        }
        index += threads;
    }
    // SAFETY: there is a partial sum for each row and each thread.
    unsafe {
        *partials.add(item) = gg;
        *partials.add(threads + item) = xx;
        for j in 0..CORRECTIONS {
            *partials.add((2 + j) * threads + item) = sg[j];
            *partials.add((2 + CORRECTIONS + j) * threads + item) = yg[j];
        }
    }
}

/// The first pass of the dot products of the update of the history, after the correction in the
/// slot `slot` is written: as [`history_dots`], for the `ROWS_UPDATE` dot products. The rows are
/// `s . y`, `s . s` and `y . y` of the new correction, then the dot products of each old step
/// with the new change, of the new step with each old change, and of each old change with the new
/// one.
///
/// # Safety
///
/// The pools must be `CORRECTIONS + 1` vectors of `count` values, `slot` and the slots in `order`
/// must be among those, and `partials` must be `ROWS_UPDATE * threads`.
pub unsafe fn update_dots(
    item: usize,
    count: usize,
    threads: usize,
    active: usize,
    order: *const u32,
    pool_s: *const f64,
    pool_y: *const f64,
    slot: usize,
    partials: *mut f64,
) {
    if item >= threads {
        return;
    }
    // SAFETY: as the caller says.
    let positions = unsafe { positions(active, order, count) };
    let new = slot * count;
    let (mut sy, mut ss, mut yy) = (0.0, 0.0, 0.0);
    let mut old_s_new_y = [0.0; CORRECTIONS];
    let mut new_s_old_y = [0.0; CORRECTIONS];
    let mut old_y_new_y = [0.0; CORRECTIONS];
    let mut index = item;
    while index < count {
        // SAFETY: the new correction is in the pools, and the index is among its values.
        let (sn, yn) = unsafe { (*pool_s.add(new + index), *pool_y.add(new + index)) };
        sy += sn * yn;
        ss += sn * sn;
        yy += yn * yn;
        for j in 0..CORRECTIONS {
            if j < active {
                // SAFETY: the correction is in the pools, and the index is among its values.
                let (so, yo) = unsafe {
                    (
                        *pool_s.add(positions[j] + index),
                        *pool_y.add(positions[j] + index),
                    )
                };
                old_s_new_y[j] += so * yn;
                new_s_old_y[j] += sn * yo;
                old_y_new_y[j] += yo * yn;
            }
        }
        index += threads;
    }
    // SAFETY: there is a partial sum for each row and each thread.
    unsafe {
        *partials.add(item) = sy;
        *partials.add(threads + item) = ss;
        *partials.add(2 * threads + item) = yy;
        for j in 0..CORRECTIONS {
            *partials.add((3 + j) * threads + item) = old_s_new_y[j];
            *partials.add((3 + CORRECTIONS + j) * threads + item) = new_s_old_y[j];
            *partials.add((3 + 2 * CORRECTIONS + j) * threads + item) = old_y_new_y[j];
        }
    }
}

/// A pass of the sums of `rows` rows of `count` values after the first: the thread `item` is a
/// group of a row, and adds up its `group` values, in order, into `sums[item]`, where the groups
/// of a row are `count.div_ceil(group)` and follow the groups of the row before.
///
/// # Safety
///
/// `values` must be `rows * count` values and `sums` must be `rows * count.div_ceil(group)`.
pub unsafe fn sum_columns(
    item: usize,
    count: usize,
    group: usize,
    rows: usize,
    values: *const f64,
    sums: *mut f64,
) {
    let groups = count.div_ceil(group);
    if item >= rows * groups {
        return;
    }
    let (row, member) = (item / groups, item % groups);
    let first = row * count + member * group;
    let last = row * count + ((member + 1) * group).min(count);
    let mut sum = 0.0;
    for index in first..last {
        // SAFETY: the index is among the values of the row.
        sum += unsafe { *values.add(index) };
    }
    // SAFETY: there is a sum for each group of each row.
    unsafe { *sums.add(item) = sum };
}

/// `d = coefficients[0] * g + sum of coefficients[1 + j] * s_j + coefficients[1 + CORRECTIONS +
/// j] * y_j`, over the corrections in use, in order, which is the direction of the compact form.
///
/// # Safety
///
/// `g` and `d` must be `count` values, `coefficients` must be `1 + 2 * CORRECTIONS`, and the
/// pools and the slots in `order` must be as for [`history_dots`].
pub unsafe fn combine(
    item: usize,
    count: usize,
    active: usize,
    order: *const u32,
    coefficients: *const f64,
    pool_s: *const f64,
    pool_y: *const f64,
    g: *const f64,
    d: *mut f64,
) {
    if item >= count {
        return;
    }
    // SAFETY: as the caller says.
    let positions = unsafe { positions(active, order, count) };
    // SAFETY: the item is among the `count` values.
    let mut value = unsafe { *coefficients * *g.add(item) };
    for j in 0..CORRECTIONS {
        if j < active {
            // SAFETY: the correction is in the pools, and the item is among its values.
            value += unsafe {
                *coefficients.add(1 + j) * *pool_s.add(positions[j] + item)
                    + *coefficients.add(1 + CORRECTIONS + j) * *pool_y.add(positions[j] + item)
            };
        }
    }
    // SAFETY: the item is among the `count` values.
    unsafe { *d.add(item) = value };
}

/// `out[offset + item] = alpha * x[item]`.
///
/// # Safety
///
/// `x` must be `count` values and `out` must have `count` values from `offset`.
pub unsafe fn scale_at(
    item: usize,
    count: usize,
    alpha: f64,
    x: *const f64,
    out: *mut f64,
    offset: usize,
) {
    if item >= count {
        return;
    }
    // SAFETY: the item is among the `count` values.
    unsafe { *out.add(offset + item) = alpha * *x.add(item) };
}

/// `out[offset + item] = new[item] - old[item]`.
///
/// # Safety
///
/// `new` and `old` must be `count` values and `out` must have `count` values from `offset`.
pub unsafe fn difference_at(
    item: usize,
    count: usize,
    new: *const f64,
    old: *const f64,
    out: *mut f64,
    offset: usize,
) {
    if item >= count {
        return;
    }
    // SAFETY: the item is among the `count` values.
    unsafe { *out.add(offset + item) = *new.add(item) - *old.add(item) };
}

// The kernels of a sweep of the fit: what is found from the points at its start and at its end.

/// The size of the thread `item`, a node: the mean length of the edges that leave it, to the nodes
/// in `entries[starts[item]..starts[item + 1]]`.
///
/// # Safety
///
/// `starts` must be `nodes + 1` values, `entries` as many as the last says, `coordinates` must be
/// `3 * nodes`, and `lengths` must be `nodes`.
pub unsafe fn node_lengths(
    item: usize,
    nodes: usize,
    starts: *const u32,
    entries: *const u32,
    coordinates: *const f64,
    lengths: *mut f64,
) {
    if item >= nodes {
        return;
    }
    // SAFETY: as the caller says.
    let (starts, coordinates) = unsafe { (view(starts, nodes + 1), view(coordinates, 3 * nodes)) };
    let (first, last) = (starts[item] as usize, starts[item + 1] as usize);
    // SAFETY: the entries are as many as the last of the starts says.
    let entries = unsafe { view(entries, last) };
    let mut sum = 0.0;
    for &other in &entries[first..last] {
        let other = other as usize;
        let d = [
            coordinates[other] - coordinates[item],
            coordinates[nodes + other] - coordinates[nodes + item],
            coordinates[2 * nodes + other] - coordinates[2 * nodes + item],
        ];
        sum += sqrt(dot(d, d));
    }
    // SAFETY: there is a length for each node.
    unsafe { *lengths.add(item) = sum / (last - first) as f64 };
}

/// The size of the thread `item`, an element: the mean of the sizes of its nodes.
///
/// # Safety
///
/// `connectivity` must be `8 * elements`, `lengths` must be `nodes`, and `scales` must be
/// `elements`.
pub unsafe fn element_scales(
    item: usize,
    elements: usize,
    nodes: usize,
    connectivity: *const u32,
    lengths: *const f64,
    scales: *mut f64,
) {
    if item >= elements {
        return;
    }
    // SAFETY: as the caller says.
    let (connectivity, lengths) =
        unsafe { (view(connectivity, 8 * elements), view(lengths, nodes)) };
    let mut sum = 0.0;
    for corner in 0..8 {
        sum += lengths[connectivity[corner * elements + item] as usize];
    }
    // SAFETY: there is a size for each element.
    unsafe { *scales.add(item) = sum / 8.0 };
}

/// The target of the thread `item`, a face of the boundary: the point of the sphere of `radius`
/// that is nearest to the centroid of the face, the normal there, and the largest of the squared
/// deviations of the nodes of the face from the plane that they make. The nodes of the face
/// `face` are `face_nodes[c * faces + face]`.
///
/// # Safety
///
/// `face_nodes` must be `4 * faces`, `coordinates` must be `3 * nodes`, and `points`, `normals`
/// must be `3 * faces` and `distances` must be `faces`.
pub unsafe fn face_targets(
    item: usize,
    faces: usize,
    nodes: usize,
    face_nodes: *const u32,
    coordinates: *const f64,
    radius: f64,
    points: *mut f64,
    normals: *mut f64,
    distances: *mut f64,
) {
    if item >= faces {
        return;
    }
    // SAFETY: as the caller says.
    let (face_nodes, coordinates) =
        unsafe { (view(face_nodes, 4 * faces), view(coordinates, 3 * nodes)) };
    let mut centroid = [0.0; 3];
    for c in 0..4 {
        let node = face_nodes[c * faces + item] as usize;
        for d in 0..3 {
            centroid[d] += coordinates[d * nodes + node] * 0.25;
        }
    }
    let norm = sqrt(dot(centroid, centroid));
    let normal = centroid.map(|c| c / norm);
    let point = normal.map(|c| c * radius);
    let mut distance = 0.0_f64;
    for c in 0..4 {
        let node = face_nodes[c * faces + item] as usize;
        let at = [
            coordinates[node],
            coordinates[nodes + node],
            coordinates[2 * nodes + node],
        ];
        let deviation = dot(minus(at, point), normal);
        distance = distance.max(deviation * deviation);
    }
    for d in 0..3 {
        // SAFETY: there is a point and a normal for each face.
        unsafe {
            *points.add(d * faces + item) = point[d];
            *normals.add(d * faces + item) = normal[d];
        }
    }
    // SAFETY: there is a distance for each face.
    unsafe { *distances.add(item) = distance };
}

/// The determinant of the corner `C`, with the nodes `A0`, `A1` and `A2` next to it.
#[inline(always)]
fn corner_determinant<const C: usize, const A0: usize, const A1: usize, const A2: usize>(
    points: &[Point; 8],
) -> f64 {
    corner(points, C, &[A0, A1, A2]).2
}

/// The least determinant of a corner of the element of the thread `item`, over the cube of its
/// size, into `out[item]`.
///
/// # Safety
///
/// The arrays must be as the layout says and `out` must be `elements`.
pub unsafe fn element_worst(
    item: usize,
    elements: usize,
    nodes: usize,
    connectivity: *const u32,
    coordinates: *const f64,
    scales: *const f64,
    out: *mut f64,
) {
    if item >= elements {
        return;
    }
    // SAFETY: as the caller says.
    let (points, scales) = unsafe {
        (
            points(item, elements, nodes, connectivity, coordinates),
            view(scales, elements),
        )
    };
    let scale = scales[item];
    let mut least = f64::INFINITY;
    // The corners of a hexahedron, each with the three nodes that it is joined to.
    least = least.min(corner_determinant::<0, 1, 3, 4>(&points));
    least = least.min(corner_determinant::<1, 2, 0, 5>(&points));
    least = least.min(corner_determinant::<2, 3, 1, 6>(&points));
    least = least.min(corner_determinant::<3, 0, 2, 7>(&points));
    least = least.min(corner_determinant::<4, 7, 5, 0>(&points));
    least = least.min(corner_determinant::<5, 4, 6, 1>(&points));
    least = least.min(corner_determinant::<6, 5, 7, 2>(&points));
    least = least.min(corner_determinant::<7, 6, 4, 3>(&points));
    // SAFETY: there is a value for each element.
    unsafe { *out.add(item) = least / (scale * scale * scale) };
}

/// Minus how far the node of the thread `item` is from where it was, over its size, so that the
/// greatest shift is the least of these.
///
/// # Safety
///
/// `start` and `x` must be `3 * nodes`, and `lengths` and `out` must be `nodes`.
pub unsafe fn shift_values(
    item: usize,
    nodes: usize,
    start: *const f64,
    x: *const f64,
    lengths: *const f64,
    out: *mut f64,
) {
    if item >= nodes {
        return;
    }
    // SAFETY: as the caller says.
    let (start, x, lengths) = unsafe {
        (
            view(start, 3 * nodes),
            view(x, 3 * nodes),
            view(lengths, nodes),
        )
    };
    let d = [
        x[item] - start[item],
        x[nodes + item] - start[nodes + item],
        x[2 * nodes + item] - start[2 * nodes + item],
    ];
    // SAFETY: there is a value for each node.
    unsafe { *out.add(item) = -(sqrt(dot(d, d)) / lengths[item]) };
}

/// The first pass of the least of `count` values, as [`sum_strided`] is of the sum.
///
/// # Safety
///
/// `values` must be `count` values and `partials` must be `threads`.
pub unsafe fn min_strided(
    item: usize,
    count: usize,
    threads: usize,
    values: *const f64,
    partials: *mut f64,
) {
    if item >= threads {
        return;
    }
    let mut least = f64::INFINITY;
    let mut index = item;
    while index < count {
        // SAFETY: the index is among the `count` values.
        least = least.min(unsafe { *values.add(index) });
        index += threads;
    }
    // SAFETY: there is a partial value for each thread.
    unsafe { *partials.add(item) = least };
}

/// A pass of the least values of `rows` rows after the first, as [`sum_columns`] is of the sums.
///
/// # Safety
///
/// `values` must be `rows * count` values and `mins` must be `rows * count.div_ceil(group)`.
pub unsafe fn min_columns(
    item: usize,
    count: usize,
    group: usize,
    rows: usize,
    values: *const f64,
    mins: *mut f64,
) {
    let groups = count.div_ceil(group);
    if item >= rows * groups {
        return;
    }
    let (row, member) = (item / groups, item % groups);
    let first = row * count + member * group;
    let last = row * count + ((member + 1) * group).min(count);
    let mut least = f64::INFINITY;
    for index in first..last {
        // SAFETY: the index is among the values of the row.
        least = least.min(unsafe { *values.add(index) });
    }
    // SAFETY: there is a value for each group of each row.
    unsafe { *mins.add(item) = least };
}
