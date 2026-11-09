//! The kernels of the fit benchmark, for a CUDA device.
//!
//! This is the `fit_kernels` example of the library, which `fit.sh` builds with the nightly
//! toolchain for `nvptx64-nvidia-cuda` into PTX that the driver compiles for whatever device it
//! runs on. It uses nothing of the library: the arithmetic is in `fit_shared.rs`, which the
//! benchmark also runs on the host, and each kernel here is that for the thread that it is
//! launched as.

#![no_std]
#![feature(abi_ptx, stdarch_nvptx, core_float_math)]
// A kernel takes its arrays and its sizes as arguments.
#![allow(clippy::too_many_arguments)]

use core::{arch::nvptx, panic::PanicInfo};

#[path = "fit_shared.rs"]
mod shared;

fn sqrt(x: f64) -> f64 {
    core::f64::math::sqrt(x)
}

#[panic_handler]
fn panic(_: &PanicInfo) -> ! {
    // SAFETY: stops the thread.
    unsafe { nvptx::trap() }
}

/// The index of the thread among all of the threads of the launch.
fn thread() -> usize {
    // SAFETY: these read registers of the launch.
    unsafe {
        (nvptx::_block_idx_x() as usize) * (nvptx::_block_dim_x() as usize)
            + nvptx::_thread_idx_x() as usize
    }
}

/// See [`shared::corner_contributions`].
///
/// # Safety
///
/// The arrays must be as the layout says.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn corner_contributions(
    elements: u32,
    nodes: u32,
    connectivity: *const u32,
    coordinates: *const f64,
    scales: *const f64,
    epsilon: f64,
    contributions: *mut f64,
) {
    // SAFETY: as the caller says.
    unsafe {
        shared::corner_contributions(
            thread(),
            elements as usize,
            nodes as usize,
            connectivity,
            coordinates,
            scales,
            epsilon,
            contributions,
        )
    }
}

/// See [`shared::element_energies`].
///
/// # Safety
///
/// The arrays must be as the layout says.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn element_energies(
    elements: u32,
    nodes: u32,
    connectivity: *const u32,
    coordinates: *const f64,
    scales: *const f64,
    epsilon: f64,
    energies: *mut f64,
) {
    // SAFETY: as the caller says.
    unsafe {
        shared::element_energies(
            thread(),
            elements as usize,
            nodes as usize,
            connectivity,
            coordinates,
            scales,
            epsilon,
            energies,
        )
    }
}

/// See [`shared::node_energies`].
///
/// # Safety
///
/// The arrays must be as the layout says.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn node_energies(
    nodes: u32,
    offset: u32,
    face_starts: *const u32,
    face_entries: *const u32,
    faces: u32,
    points: *const f64,
    normals: *const f64,
    distances: *const f64,
    coordinates: *const f64,
    lengths: *const f64,
    energies: *mut f64,
) {
    // SAFETY: as the caller says.
    unsafe {
        shared::node_energies(
            thread(),
            nodes as usize,
            offset as usize,
            face_starts,
            face_entries,
            faces as usize,
            points,
            normals,
            distances,
            coordinates,
            lengths,
            energies,
        )
    }
}

/// See [`shared::gradient`].
///
/// # Safety
///
/// The arrays must be as the layout says.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn gradient(
    nodes: u32,
    elements: u32,
    starts: *const u32,
    entries: *const u32,
    contributions: *const f64,
    face_starts: *const u32,
    face_entries: *const u32,
    faces: u32,
    points: *const f64,
    normals: *const f64,
    distances: *const f64,
    coordinates: *const f64,
    lengths: *const f64,
    gradient: *mut f64,
) {
    // SAFETY: as the caller says.
    unsafe {
        shared::gradient(
            thread(),
            nodes as usize,
            elements as usize,
            starts,
            entries,
            contributions,
            face_starts,
            face_entries,
            faces as usize,
            points,
            normals,
            distances,
            coordinates,
            lengths,
            gradient,
        )
    }
}

/// See [`shared::sum_strided`].
///
/// # Safety
///
/// `values` must be `count` values and `partials` must be `threads`.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn sum_strided(
    count: u32,
    threads: u32,
    values: *const f64,
    partials: *mut f64,
) {
    // SAFETY: as the caller says.
    unsafe { shared::sum_strided(thread(), count as usize, threads as usize, values, partials) }
}

/// See [`shared::sum_groups`].
///
/// # Safety
///
/// `values` must be `count` values and `sums` must be `count.div_ceil(group)`.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn sum_groups(
    count: u32,
    group: u32,
    values: *const f64,
    sums: *mut f64,
) {
    // SAFETY: as the caller says.
    unsafe { shared::sum_groups(thread(), count as usize, group as usize, values, sums) }
}

/// See [`shared::dot_strided`].
///
/// # Safety
///
/// `a` and `b` must be `count` values and `partials` must be `threads`.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn dot_strided(
    count: u32,
    threads: u32,
    a: *const f64,
    b: *const f64,
    partials: *mut f64,
) {
    // SAFETY: as the caller says.
    unsafe { shared::dot_strided(thread(), count as usize, threads as usize, a, b, partials) }
}

/// See [`shared::sum_into`].
///
/// # Safety
///
/// `values` must be `count` values and `out` must have a value at `slot`.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn sum_into(
    count: u32,
    values: *const f64,
    out: *mut f64,
    slot: u32,
) {
    // SAFETY: as the caller says.
    unsafe { shared::sum_into(thread(), count as usize, values, out, slot as usize) }
}

/// See [`shared::times`].
///
/// # Safety
///
/// `dots` must have a value at `dot` and `out` must have a value at `index`.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn times(
    factor: f64,
    dots: *const f64,
    dot: u32,
    out: *mut f64,
    index: u32,
) {
    // SAFETY: as the caller says.
    unsafe { shared::times(thread(), factor, dots, dot as usize, out, index as usize) }
}

/// See [`shared::axpy`].
///
/// # Safety
///
/// `x`, `y` and `out` must be `count` values.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn axpy(
    count: u32,
    alpha: f64,
    x: *const f64,
    y: *const f64,
    out: *mut f64,
) {
    // SAFETY: as the caller says.
    unsafe { shared::axpy(thread(), count as usize, alpha, x, y, out) }
}

/// See [`shared::scale`].
///
/// # Safety
///
/// `x` and `out` must be `count` values.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn scale(count: u32, alpha: f64, x: *const f64, out: *mut f64) {
    // SAFETY: as the caller says.
    unsafe { shared::scale(thread(), count as usize, alpha, x, out) }
}

/// See [`shared::scale_in_place`].
///
/// # Safety
///
/// `y` must be `count` values.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn scale_in_place(count: u32, alpha: f64, y: *mut f64) {
    // SAFETY: as the caller says.
    unsafe { shared::scale_in_place(thread(), count as usize, alpha, y) }
}

/// See [`shared::subtract_alpha`].
///
/// # Safety
///
/// `x` and `y` must be `count` values and `scalars` must have a value at `index`.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn subtract_alpha(
    count: u32,
    scalars: *const f64,
    index: u32,
    x: *const f64,
    y: *mut f64,
) {
    // SAFETY: as the caller says.
    unsafe { shared::subtract_alpha(thread(), count as usize, scalars, index as usize, x, y) }
}

/// See [`shared::subtract_pair`].
///
/// # Safety
///
/// `x` and `y` must be `count` values and `alphas` and `betas` must have a value at `index`.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn subtract_pair(
    count: u32,
    alphas: *const f64,
    betas: *const f64,
    index: u32,
    x: *const f64,
    y: *mut f64,
) {
    // SAFETY: as the caller says.
    unsafe {
        shared::subtract_pair(
            thread(),
            count as usize,
            alphas,
            betas,
            index as usize,
            x,
            y,
        )
    }
}

/// See [`shared::history_dots`].
///
/// # Safety
///
/// The arrays must be as the layout says.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn history_dots(
    count: u32,
    threads: u32,
    active: u32,
    order: *const u32,
    pool_s: *const f64,
    pool_y: *const f64,
    g: *const f64,
    x: *const f64,
    partials: *mut f64,
) {
    // SAFETY: as the caller says.
    unsafe {
        shared::history_dots(
            thread(),
            count as usize,
            threads as usize,
            active as usize,
            order,
            pool_s,
            pool_y,
            g,
            x,
            partials,
        )
    }
}

/// See [`shared::update_dots`].
///
/// # Safety
///
/// The arrays must be as the layout says.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn update_dots(
    count: u32,
    threads: u32,
    active: u32,
    order: *const u32,
    pool_s: *const f64,
    pool_y: *const f64,
    slot: u32,
    partials: *mut f64,
) {
    // SAFETY: as the caller says.
    unsafe {
        shared::update_dots(
            thread(),
            count as usize,
            threads as usize,
            active as usize,
            order,
            pool_s,
            pool_y,
            slot as usize,
            partials,
        )
    }
}

/// See [`shared::sum_columns`].
///
/// # Safety
///
/// `values` must be `rows * count` values and `sums` must be `rows * count.div_ceil(group)`.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn sum_columns(
    count: u32,
    group: u32,
    rows: u32,
    values: *const f64,
    sums: *mut f64,
) {
    // SAFETY: as the caller says.
    unsafe {
        shared::sum_columns(
            thread(),
            count as usize,
            group as usize,
            rows as usize,
            values,
            sums,
        )
    }
}

/// See [`shared::combine`].
///
/// # Safety
///
/// The arrays must be as the layout says.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn combine(
    count: u32,
    active: u32,
    order: *const u32,
    coefficients: *const f64,
    pool_s: *const f64,
    pool_y: *const f64,
    g: *const f64,
    d: *mut f64,
) {
    // SAFETY: as the caller says.
    unsafe {
        shared::combine(
            thread(),
            count as usize,
            active as usize,
            order,
            coefficients,
            pool_s,
            pool_y,
            g,
            d,
        )
    }
}

/// See [`shared::scale_at`].
///
/// # Safety
///
/// `x` must be `count` values and `out` must have `count` values from `offset`.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn scale_at(
    count: u32,
    alpha: f64,
    x: *const f64,
    out: *mut f64,
    offset: u32,
) {
    // SAFETY: as the caller says.
    unsafe { shared::scale_at(thread(), count as usize, alpha, x, out, offset as usize) }
}

/// See [`shared::difference_at`].
///
/// # Safety
///
/// `new` and `old` must be `count` values and `out` must have `count` values from `offset`.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn difference_at(
    count: u32,
    new: *const f64,
    old: *const f64,
    out: *mut f64,
    offset: u32,
) {
    // SAFETY: as the caller says.
    unsafe { shared::difference_at(thread(), count as usize, new, old, out, offset as usize) }
}

/// See [`shared::node_lengths`].
///
/// # Safety
///
/// The arrays must be as the layout says.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn node_lengths(
    nodes: u32,
    starts: *const u32,
    entries: *const u32,
    coordinates: *const f64,
    lengths: *mut f64,
) {
    // SAFETY: as the caller says.
    unsafe {
        shared::node_lengths(
            thread(),
            nodes as usize,
            starts,
            entries,
            coordinates,
            lengths,
        )
    }
}

/// See [`shared::element_scales`].
///
/// # Safety
///
/// The arrays must be as the layout says.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn element_scales(
    elements: u32,
    nodes: u32,
    connectivity: *const u32,
    lengths: *const f64,
    scales: *mut f64,
) {
    // SAFETY: as the caller says.
    unsafe {
        shared::element_scales(
            thread(),
            elements as usize,
            nodes as usize,
            connectivity,
            lengths,
            scales,
        )
    }
}

/// See [`shared::face_targets`].
///
/// # Safety
///
/// The arrays must be as the layout says.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn face_targets(
    faces: u32,
    nodes: u32,
    face_nodes: *const u32,
    coordinates: *const f64,
    radius: f64,
    points: *mut f64,
    normals: *mut f64,
    distances: *mut f64,
) {
    // SAFETY: as the caller says.
    unsafe {
        shared::face_targets(
            thread(),
            faces as usize,
            nodes as usize,
            face_nodes,
            coordinates,
            radius,
            points,
            normals,
            distances,
        )
    }
}

/// See [`shared::element_worst`].
///
/// # Safety
///
/// The arrays must be as the layout says.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn element_worst(
    elements: u32,
    nodes: u32,
    connectivity: *const u32,
    coordinates: *const f64,
    scales: *const f64,
    out: *mut f64,
) {
    // SAFETY: as the caller says.
    unsafe {
        shared::element_worst(
            thread(),
            elements as usize,
            nodes as usize,
            connectivity,
            coordinates,
            scales,
            out,
        )
    }
}

/// See [`shared::shift_values`].
///
/// # Safety
///
/// `start` and `x` must be `3 * nodes`, and `lengths` and `out` must be `nodes`.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn shift_values(
    nodes: u32,
    start: *const f64,
    x: *const f64,
    lengths: *const f64,
    out: *mut f64,
) {
    // SAFETY: as the caller says.
    unsafe { shared::shift_values(thread(), nodes as usize, start, x, lengths, out) }
}

/// See [`shared::min_strided`].
///
/// # Safety
///
/// `values` must be `count` values and `partials` must be `threads`.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn min_strided(
    count: u32,
    threads: u32,
    values: *const f64,
    partials: *mut f64,
) {
    // SAFETY: as the caller says.
    unsafe { shared::min_strided(thread(), count as usize, threads as usize, values, partials) }
}

/// See [`shared::min_columns`].
///
/// # Safety
///
/// `values` must be `rows * count` values and `mins` must be `rows * count.div_ceil(group)`.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn min_columns(
    count: u32,
    group: u32,
    rows: u32,
    values: *const f64,
    mins: *mut f64,
) {
    // SAFETY: as the caller says.
    unsafe {
        shared::min_columns(
            thread(),
            count as usize,
            group as usize,
            rows as usize,
            values,
            mins,
        )
    }
}

/// See [`shared::surface_targets`].
///
/// # Safety
///
/// The arrays must be as the layout and the surface say.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn surface_targets(
    faces: u32,
    nodes: u32,
    tree_nodes: u32,
    triangle_count: u32,
    face_nodes: *const u32,
    coordinates: *const f64,
    bounds: *const f64,
    skips: *const u32,
    firsts: *const u32,
    counts: *const u32,
    triangles: *const f64,
    previous: *mut u32,
    points: *mut f64,
    normals: *mut f64,
    distances: *mut f64,
) {
    // SAFETY: as the caller says.
    unsafe {
        shared::surface_targets(
            thread(),
            faces as usize,
            nodes as usize,
            tree_nodes as usize,
            triangle_count as usize,
            face_nodes,
            coordinates,
            bounds,
            skips,
            firsts,
            counts,
            triangles,
            previous,
            points,
            normals,
            distances,
        )
    }
}
