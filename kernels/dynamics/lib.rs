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
