#![no_std]
#![feature(abi_ptx, stdarch_nvptx)]
#![allow(clippy::too_many_arguments)]

use conspire::{
    constitutive::{
        FromParameters,
        solid::{
            elastic::Elastic,
            hyperelastic::{NeoHookean, SaintVenantKirchhoff},
        },
    },
    fem::element::{
        ErrorRecord, Recomputed, StandardGradientOperators, linear::LinearElement,
        solid::elastic::ElasticFiniteElement,
    },
    math::{ScalarList, Tensor},
};
use core::{
    alloc::{GlobalAlloc, Layout},
    arch::nvptx::{_block_dim_x, _block_idx_x, _thread_idx_x, trap},
    panic::PanicInfo,
    slice::from_raw_parts,
    sync::atomic::{AtomicU64, Ordering},
};

#[panic_handler]
fn panic(_: &PanicInfo) -> ! {
    // SAFETY: stops the thread.
    unsafe { trap() }
}

struct NoHeap;

// SAFETY: it never gives any memory.
unsafe impl GlobalAlloc for NoHeap {
    unsafe fn alloc(&self, _: Layout) -> *mut u8 {
        // SAFETY: stops the thread.
        unsafe { trap() }
    }
    unsafe fn dealloc(&self, _: *mut u8, _: Layout) {}
}

#[global_allocator]
static HEAP: NoHeap = NoHeap;

fn thread() -> usize {
    // SAFETY: these read registers of the launch.
    unsafe { (_block_idx_x() as usize) * (_block_dim_x() as usize) + _thread_idx_x() as usize }
}

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

/// The nodal forces of each of `items` elements of a block, one element to a thread, with the
/// gradient vectors and weights recomputed from the reference coordinates of the nodes.
///
/// The node `a` of an element `item` is `connectivity[a * items + item]`, and the component `i`
/// of the force on it is `forces[(i * N + a) * items + item]`. The nodes have three coordinates,
/// as `coordinates[3 * node + i]` and `reference_coordinates[3 * node + i]`. The elements share
/// the `derivatives`, as `[point][node][component]`, and the `weights`, as `[point]`.
///
/// An element whose stress is an error has forces of zeros, and `record`, which the host sets to
/// [`ErrorRecord::NONE`], takes the least of the records of the errors that the elements have.
///
/// # Safety
///
/// `reference_coordinates` and `coordinates` must have the nodes that `connectivity`, which must
/// be `N * items` of them, names, `derivatives` must be `G * N * 3` values and `weights` `G`,
/// `forces` must be `3 * N * items` values, `record` must be one aligned value, and `parameters`
/// must be `number_of_parameters` values that make a model.
#[inline(always)]
unsafe fn block_forces<M, const G: usize, const N: usize>(
    items: u32,
    reference_coordinates: *const f64,
    derivatives: *const f64,
    weights: *const f64,
    connectivity: *const u32,
    coordinates: *const f64,
    forces: *mut f64,
    record: *mut u64,
    parameters: *const f64,
    number_of_parameters: u32,
) where
    M: Elastic + FromParameters,
    LinearElement<G, N>: ElasticFiniteElement<M, G, 3, N, N>,
{
    let item = thread();
    let items = items as usize;
    if item < items {
        // SAFETY: there are as many parameters as the model takes.
        let model = M::from_parameters(unsafe {
            from_raw_parts(parameters, number_of_parameters as usize)
        });
        // SAFETY: the tables are those of the parametric element, which are as large as these,
        // and the tensors are transparent over the numbers that they are made of.
        let (derivatives, weights) = unsafe {
            (
                &*derivatives.cast::<StandardGradientOperators<3, N, G>>(),
                &*weights.cast::<ScalarList<G>>(),
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
                position
                    .iter_mut()
                    .zip(node_reference.iter_mut())
                    .enumerate()
                    .for_each(|(i, (position, node_reference))| {
                        // SAFETY: the node has three coordinates in each of the arrays.
                        *position = unsafe { *coordinates.add(3 * node + i) };
                        // SAFETY: as above.
                        *node_reference = unsafe { *reference_coordinates.add(3 * node + i) };
                    });
            });
        let element = Recomputed::new(nodes.into(), derivatives, weights);
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
                // SAFETY: `record` is one aligned value, which the threads only take the least.
                unsafe { AtomicU64::from_ptr(record) }
                    .fetch_min(ErrorRecord::of(item, &error).bits(), Ordering::Relaxed);
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
/// `$nodes` nodes, named `block_forces_<name>_g<points>_n<nodes>`,
/// which is how the host finds the kernel of a block.
macro_rules! kernel {
    ($model:ty, $name:literal, $points:literal, $nodes:literal) => {
        const _: () = {
            assert!(
                same(<$model as FromParameters>::NAME, $name),
                "the name of the kernels is not the name of the model"
            );
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
                record: *mut u64,
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
                        record,
                        parameters,
                        number_of_parameters,
                    )
                }
            }
        };
    };
}

/// The kernels of every model for every shape of element.
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
