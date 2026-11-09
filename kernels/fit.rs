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
