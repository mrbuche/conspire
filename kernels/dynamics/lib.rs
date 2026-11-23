#![no_std]
#![feature(abi_ptx, stdarch_nvptx)]
#![allow(clippy::too_many_arguments)]

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

/// The accelerations that balance the forces on `count` degrees of freedom, three to a node, as
/// `-force / mass` with the mass of the node, and zero for those that are held; and, if `kick` is
/// not zero, the velocity another half step on with them, which ends a step.
///
/// The `loads` external forces are subtracted from the force on a degree of freedom, one at a
/// time in the order that they are given, as the host subtracts them from the forces of the
/// elements: the load `l` is a force on each degree of freedom, as `data[l][dof]`, if `kinds[l]`
/// is zero, and otherwise a weight of each node, as `data[l][node]`, times the component of the
/// row `row` of its table, as `tables[l][3 * row + component]`.
///
/// # Safety
///
/// `forces`, `held`, `accelerations` and `velocities` must each be `count` values, and `masses`
/// the masses of the nodes, `count / 3` of them. `kinds`, `data` and `tables` must each be
/// `loads` values, the addresses in `data` and `tables` must be those of arrays of the values
/// that the kind of the load says, and of the row `row` of three values for a weighted load,
/// which `tables` need not have for a load that does not change.
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
    loads: u32,
    kinds: *const u32,
    data: *const u64,
    tables: *const u64,
    row: u32,
) {
    let dof = thread();
    if dof < count as usize {
        // SAFETY: the degree of freedom is one that the arrays have, its node has a mass, and the
        // loads are those that the safety of the kernel says.
        unsafe {
            let acceleration = if *held.add(dof) != 0 {
                0.0
            } else {
                let mut force = *forces.add(dof);
                for load in 0..loads as usize {
                    let values = *data.add(load) as *const f64;
                    if *kinds.add(load) == 0 {
                        force -= *values.add(dof);
                    } else {
                        let table = *tables.add(load) as *const f64;
                        let component = *table.add(row as usize * 3 + dof % 3);
                        force -= *values.add(dof / 3) * component;
                    }
                }
                -force / *masses.add(dof / 3)
            };
            *accelerations.add(dof) = acceleration;
            if kick != 0 {
                *velocities.add(dof) = acceleration * (dt * 0.5) + *velocities.add(dof);
            }
        }
    }
}
