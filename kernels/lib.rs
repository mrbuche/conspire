//! conspire's own code, compiled for a CUDA device.
//!
//! This is the `kernels` example of the library, which `build.sh` builds with the nightly
//! toolchain for `nvptx64-nvidia-cuda` into PTX that the driver compiles for whatever device it
//! runs on.

#![no_std]
#![feature(abi_ptx, stdarch_nvptx)]
// A kernel takes its arrays and its sizes as arguments.
#![allow(clippy::too_many_arguments)]

use conspire::{
    constitutive::solid::{
        elastic::DeviceModel,
        hyperelastic::{NeoHookean, SaintVenantKirchhoff},
    },
    fem::element::{
        Failure, OnTheFly, linear::LinearElement, solid::elastic::ElasticFiniteElement,
    },
    math::Tensor,
};
use core::{
    alloc::{GlobalAlloc, Layout},
    arch::nvptx,
    panic::PanicInfo,
    slice::from_raw_parts,
    sync::atomic::{AtomicU64, Ordering},
};

#[panic_handler]
fn panic(_: &PanicInfo) -> ! {
    // SAFETY: stops the thread.
    unsafe { nvptx::trap() }
}

/// A device has no heap that the code here uses, so asking for memory stops the thread.
struct NoHeap;

// SAFETY: it never gives any memory.
unsafe impl GlobalAlloc for NoHeap {
    unsafe fn alloc(&self, _: Layout) -> *mut u8 {
        // SAFETY: stops the thread.
        unsafe { nvptx::trap() }
    }
    unsafe fn dealloc(&self, _: *mut u8, _: Layout) {}
}

#[global_allocator]
static HEAP: NoHeap = NoHeap;

/// The index of the thread among all of the threads of the launch.
fn thread() -> usize {
    // SAFETY: these read registers of the launch.
    unsafe {
        (nvptx::_block_idx_x() as usize) * (nvptx::_block_dim_x() as usize)
            + nvptx::_thread_idx_x() as usize
    }
}

/// Fills `output` with the squares of the numbers from one, to check that code made here runs.
///
/// # Safety
///
/// `output` must have `count` values.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn squares(output: *mut f64, count: u32) {
    let index = thread();
    if index < count as usize {
        let value = (index + 1) as f64;
        // SAFETY: the index is within the values that `output` has.
        unsafe { *output.add(index) = value * value };
    }
}

/// The nodal forces of each of `items` elements of a block, by the code of the library, one
/// element to a thread, with the gradient vectors and weights made as they are needed from the
/// reference coordinates of the nodes (see [`OnTheFly`]).
///
/// The elements are kept in arrays with the element last, so that the threads of neighboring
/// elements read from neighboring places: the node `a` of an element `item` is
/// `connectivity[a * items + item]`, and the component `i` of the force on it is
/// `forces[(i * N + a) * items + item]`. The nodes have three coordinates, as
/// `coordinates[3 * node + i]` and `reference_coordinates[3 * node + i]`. The elements of a
/// block all have the same parametric element, so they share its `derivatives`, as
/// `[point][node][component]`, and its `weights`, as `[point]`.
///
/// An element whose stress is an error has forces of zeros, and `failure`, which the host sets
/// to [`Failure::NONE`], takes the least of the failures that the elements have.
///
/// # Safety
///
/// `reference_coordinates` and `coordinates` must have the nodes that `connectivity`, which must
/// be `N * items` of them, names, `derivatives` must be `G * N * 3` values and `weights` `G`,
/// `forces` must be `3 * N * items` values, `failure` must be one value, and `parameters` must be
/// `number_of_parameters` values that make a model.
#[inline(always)]
unsafe fn block_forces<M, const G: usize, const N: usize>(
    items: u32,
    reference_coordinates: *const f64,
    derivatives: *const f64,
    weights: *const f64,
    connectivity: *const u32,
    coordinates: *const f64,
    forces: *mut f64,
    failure: *mut u64,
    parameters: *const f64,
    number_of_parameters: u32,
) where
    M: DeviceModel,
    LinearElement<G, N>: ElasticFiniteElement<M, G, 3, N, N>,
{
    let item = thread();
    let items = items as usize;
    if item < items {
        // SAFETY: there are as many parameters as the model takes.
        let model = M::from_parameters(unsafe {
            from_raw_parts(parameters, number_of_parameters as usize)
        });
        // SAFETY: the tables are those of the parametric element, which are as large as these.
        let (derivatives, weights) = unsafe {
            (
                &*derivatives.cast::<[[[f64; 3]; N]; G]>(),
                &*weights.cast::<[f64; G]>(),
            )
        };
        let mut positions = [[0.0; 3]; N];
        let mut nodes = [[0.0; 3]; N];
        positions
            .iter_mut()
            .zip(nodes.iter_mut())
            .enumerate()
            .for_each(|(a, (position, node_reference))| {
                // SAFETY: the node is one that the coordinates have.
                let node = unsafe { *connectivity.add(a * items + item) } as usize;
                (0..3).for_each(|i| unsafe {
                    position[i] = *coordinates.add(3 * node + i);
                    node_reference[i] = *reference_coordinates.add(3 * node + i);
                });
            });
        let element = OnTheFly::new(nodes, derivatives, weights);
        let mut nodal_forces = [[0.0; 3]; N];
        match <LinearElement<G, N> as ElasticFiniteElement<M, G, 3, N, N>>::nodal_forces_from(
            &element,
            &model,
            &positions.into(),
        ) {
            Ok(computed) => computed.iter().enumerate().for_each(|(a, force)| {
                force
                    .iter()
                    .enumerate()
                    .for_each(|(i, component)| nodal_forces[a][i] = component.value())
            }),
            Err(error) => {
                // SAFETY: `failure` is one value, which the threads only take the least of.
                unsafe { &*(failure as *const AtomicU64) }
                    .fetch_min(Failure::of(item, &error).bits(), Ordering::Relaxed);
            }
        }
        nodal_forces.iter().enumerate().for_each(|(a, force)| {
            force.iter().enumerate().for_each(|(i, &component)| {
                // SAFETY: the index is within the forces that there are.
                unsafe { *forces.add((i * N + a) * items + item) = component }
            })
        });
    }
}

/// The second pass of the forces: the forces on `nodes` nodes, each the sum of the forces of the
/// elements that have it, in the order of the elements, added to the forces that it has.
///
/// A thread is a node. The contributions to the node `node` are `contributions[offsets[node]..
/// offsets[node + 1]]`, in the order of the elements, and each is where the force of an element
/// is in each of the three planes of components of `forces`, which are `plane` values apart: the
/// node `a` of an element `item` is `a * items + item`, as in [`block_forces`]. The forces on the
/// nodes are `nodal_forces[3 * node + i]`.
///
/// The sum is only additions, in the order of the elements, so the forces are those that the
/// host adds, whether the operations are fused or not.
///
/// # Safety
///
/// `offsets` must be `nodes + 1` values and `contributions` what they say, each within a plane,
/// `forces` must be three planes of `plane` values, and `nodal_forces` must be `3 * nodes` values,
/// at least.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn gather(
    nodes: u32,
    plane: u32,
    offsets: *const u32,
    contributions: *const u32,
    forces: *const f64,
    nodal_forces: *mut f64,
) {
    let node = thread();
    if node < nodes as usize {
        let plane = plane as usize;
        // SAFETY: the offsets are those of the nodes, and the node is one of them.
        let (first, last, mut sum) = unsafe {
            (
                *offsets.add(node),
                *offsets.add(node + 1),
                [
                    *nodal_forces.add(3 * node),
                    *nodal_forces.add(3 * node + 1),
                    *nodal_forces.add(3 * node + 2),
                ],
            )
        };
        for k in first..last {
            // SAFETY: the contribution is within the forces of the elements.
            let contribution = unsafe { *contributions.add(k as usize) } as usize;
            sum.iter_mut()
                .enumerate()
                .for_each(|(i, value)| *value += unsafe { *forces.add(i * plane + contribution) });
        }
        sum.iter()
            .enumerate()
            .for_each(|(i, &value)| unsafe { *nodal_forces.add(3 * node + i) = value });
    }
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

const fn same(a: &str, b: &str) -> bool {
    let (a, b) = (a.as_bytes(), b.as_bytes());
    if a.len() != b.len() {
        return false;
    }
    let mut index = 0;
    while index < a.len() {
        if a[index] != b[index] {
            return false;
        }
        index += 1;
    }
    true
}

/// A kernel for the elements of a block of a model, with `$points` integration points and
/// `$nodes` nodes, named `block_forces_<name>_g<points>_n<nodes>`, which is how the host finds
/// the kernel of a block. See [`block_forces`].
macro_rules! kernel {
    ($model:ty, $name:literal, $points:literal, $nodes:literal) => {
        const _: () = {
            assert!(
                same(<$model as DeviceModel>::NAME, $name),
                "the name of the kernels is not the name of the model"
            );
            /// The nodal forces of the elements of a block, as for [`block_forces`].
            ///
            /// # Safety
            ///
            /// As for [`block_forces`].
            #[unsafe(export_name = concat!("block_forces_", $name, "_g", $points, "_n", $nodes))]
            pub unsafe extern "ptx-kernel" fn kernel(
                items: u32,
                reference_coordinates: *const f64,
                derivatives: *const f64,
                weights: *const f64,
                connectivity: *const u32,
                coordinates: *const f64,
                forces: *mut f64,
                failure: *mut u64,
                parameters: *const f64,
                number_of_parameters: u32,
            ) {
                // SAFETY: as for the function that this calls.
                unsafe {
                    block_forces::<$model, $points, $nodes>(
                        items,
                        reference_coordinates,
                        derivatives,
                        weights,
                        connectivity,
                        coordinates,
                        forces,
                        failure,
                        parameters,
                        number_of_parameters,
                    )
                }
            }
        };
    };
}

/// The kernels of every model for every shape of element, which are the shapes of the elements
/// of the library with a Jacobian that is made of the reference coordinates: the tetrahedron with
/// one point and with four, the hexahedron, the wedge, and the pyramid.
///
/// Adding a model is a line of its type and its name, which must be that of its [`DeviceModel`].
macro_rules! kernels {
    (models: [$($model:ty => $name:literal),* $(,)?], shapes: $shapes:tt $(,)?) => {
        $( kernels!(@shapes $model, $name, $shapes); )*
    };
    (@shapes $model:ty, $name:literal, [$(($points:literal, $nodes:literal)),* $(,)?]) => {
        $( kernel!($model, $name, $points, $nodes); )*
    };
}

kernels! {
    models: [
        NeoHookean => "neo_hookean",
        SaintVenantKirchhoff => "saint_venant_kirchhoff",
    ],
    shapes: [(1, 4), (4, 4), (8, 8), (6, 6), (8, 5)],
}
