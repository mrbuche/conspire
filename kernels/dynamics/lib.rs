#![no_std]
#![feature(abi_ptx, stdarch_nvptx)]

use core::{
    arch::nvptx::{_block_dim_x, _block_idx_x, _thread_idx_x, trap},
    panic::PanicInfo,
};

#[panic_handler]
fn panic(_: &PanicInfo) -> ! {
    // SAFETY: stops the thread.
    unsafe { trap() }
}

fn thread() -> usize {
    // SAFETY: these read registers of the launch.
    unsafe { (_block_idx_x() as usize) * (_block_dim_x() as usize) + _thread_idx_x() as usize }
}

/// The first half of a step of velocity Verlet on `count` degrees of freedom: the velocity a half
/// step on, `v + a dt / 2`, and with it the coordinate a step on, `x + v dt`, as the integrator
/// of the library does, with the same operations in the same order.
///
/// # Safety
///
/// `coordinates`, `velocities` and `accelerations` must each be `count` values.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn kick_drift(
    count: u32,
    coordinates: *mut f64,
    velocities: *mut f64,
    accelerations: *const f64,
    dt: f64,
) {
    let dof = thread();
    if dof < count as usize {
        // SAFETY: the degree of freedom is one that the arrays have.
        unsafe {
            let half = *accelerations.add(dof) * (dt * 0.5) + *velocities.add(dof);
            *velocities.add(dof) = half;
            *coordinates.add(dof) = half * dt + *coordinates.add(dof);
        }
    }
}

/// Sets the coordinates that are moved to what they are held at: the coordinate `dofs[i]` is set
/// to `table[row * count + i]`, for each of the `count` of them.
///
/// # Safety
///
/// `dofs` must be `count` coordinates of `coordinates`, and `table` must have the row `row` of
/// `count` values.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn set_targets(
    count: u32,
    dofs: *const u32,
    table: *const f64,
    row: u32,
    coordinates: *mut f64,
) {
    let index = thread();
    if index < count as usize {
        // SAFETY: the index is one of the `count` degrees of freedom, and the row is in the table.
        unsafe {
            let dof = *dofs.add(index) as usize;
            *coordinates.add(dof) = *table.add(row as usize * count as usize + index);
        }
    }
}

/// Subtracts a force on each of `count` degrees of freedom, which does not change, from the
/// forces on them, as the host subtracts an external force from the forces of the elements.
///
/// # Safety
///
/// `loads` and `forces` must each be `count` values.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn subtract_constant(
    count: u32,
    loads: *const f64,
    forces: *mut f64,
) {
    let dof = thread();
    if dof < count as usize {
        // SAFETY: the degree of freedom is one that the arrays have.
        unsafe { *forces.add(dof) = *forces.add(dof) - *loads.add(dof) };
    }
}

/// Subtracts a force on each of `nodes` nodes that is a weight of the node times three
/// components that are a function of time, from the forces on them: the components are
/// `table[row * 3 + c]`.
///
/// # Safety
///
/// `weights` must be `nodes` values, `forces` must be `3 * nodes` values, and `table` must have
/// the row `row` of three values.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn subtract_weighted(
    nodes: u32,
    weights: *const f64,
    table: *const f64,
    row: u32,
    forces: *mut f64,
) {
    let node = thread();
    if node < nodes as usize {
        // SAFETY: the node is one that the arrays have, and the row is in the table.
        unsafe {
            let weight = *weights.add(node);
            for c in 0..3 {
                let force = weight * *table.add(row as usize * 3 + c);
                *forces.add(3 * node + c) = *forces.add(3 * node + c) - force;
            }
        }
    }
}

/// The accelerations that balance the forces on `count` degrees of freedom, three to a node, as
/// `-force / mass` with the mass of the node, and zero for those that are held; and, if `kick` is
/// not zero, the velocity another half step on with them, which ends a step.
///
/// # Safety
///
/// `forces`, `held`, `accelerations` and `velocities` must each be `count` values, and `masses`
/// the masses of the nodes, `count / 3` of them.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn accelerate(
    count: u32,
    forces: *const f64,
    masses: *const f64,
    held: *const u32,
    accelerations: *mut f64,
    velocities: *mut f64,
    dt: f64,
    kick: u32,
) {
    let dof = thread();
    if dof < count as usize {
        // SAFETY: the degree of freedom is one that the arrays have, and its node has a mass.
        unsafe {
            let acceleration = if *held.add(dof) != 0 {
                0.0
            } else {
                -*forces.add(dof) / *masses.add(dof / 3)
            };
            *accelerations.add(dof) = acceleration;
            if kick != 0 {
                *velocities.add(dof) = acceleration * (dt * 0.5) + *velocities.add(dof);
            }
        }
    }
}
