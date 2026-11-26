#![no_std]
#![feature(abi_ptx, stdarch_nvptx)]
#![allow(clippy::too_many_arguments)]

use conspire::{
    constitutive::{
        FromParameters,
        solid::{
            elastic::{Elastic, StiffnessBound},
            hyperelastic::{NeoHookean, SaintVenantKirchhoff},
        },
    },
    domain::ModelBound,
    fem::element::{
        ElementNodalCoordinates, ErrorRecord, IntegrationData, Recomputed,
        StandardGradientOperators,
        linear::LinearElement,
        mass::ElementNodalLumpedMasses,
        solid::elastic::ElasticFiniteElement,
        stabilized::StabilizedElement,
        time_scale::{UnstableRecord, squared_frequency_bound_from},
    },
    math::{ScalarList, Tensor},
    units::Mass,
};
use core::{
    alloc::{GlobalAlloc, Layout},
    arch::nvptx::{_block_dim_x, _block_idx_x, _thread_idx_x, trap},
    array::from_fn,
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

/// The nodal forces of each of `items` elements of a block of elements of one integration point
/// with hourglass control, one element to a thread, with what each has of its own read from its
/// `values`, and the forces made as the host makes them, by the same code. The element is the
/// hexahedron of eight nodes.
///
/// The value `k` of an element `item` is `values[k * items + item]`, which are
/// [`StabilizedElement::VALUES`] of them. The nodes, the forces and the `record` are those of
/// [`block_forces`].
///
/// # Safety
///
/// `coordinates` must have the nodes that `connectivity`, which must be `8 * items` of them,
/// names, `values` must be `StabilizedElement::<8>::VALUES * items` values, `forces` must be
/// `3 * 8 * items` values, `record` must be one aligned value, and `parameters` must be
/// `number_of_parameters` values that make a model.
#[inline(always)]
unsafe fn block_forces_stabilized<M>(
    items: u32,
    values: *const f64,
    connectivity: *const u32,
    coordinates: *const f64,
    forces: *mut f64,
    record: *mut u64,
    parameters: *const f64,
    number_of_parameters: u32,
) where
    M: Elastic + FromParameters,
{
    const N: usize = 8;
    let item = thread();
    let items = items as usize;
    if item < items {
        // SAFETY: there are as many parameters as the model takes.
        let model = M::from_parameters(unsafe {
            from_raw_parts(parameters, number_of_parameters as usize)
        });
        let mut positions = [[0.0; 3]; N];
        positions.iter_mut().enumerate().for_each(|(a, position)| {
            // SAFETY: the node is one that the coordinates have.
            let node = unsafe { *connectivity.add(a * items + item) } as usize;
            position.iter_mut().enumerate().for_each(|(i, position)| {
                // SAFETY: the node has three coordinates.
                *position = unsafe { *coordinates.add(3 * node + i) }
            });
        });
        let mut nodal_forces = [[0.0; 3]; N];
        let computed = StabilizedElement::<N>::nodal_forces_from_values(
            // SAFETY: the value is one of those that the element has.
            |k| unsafe { *values.add(k * items + item) },
            &model,
            &positions.into(),
        );
        match computed {
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

/// The coordinates of the nodes of an element `item`, as [`block_forces`] has them.
///
/// # Safety
///
/// `coordinates` must have the nodes that `connectivity`, which must be `N * items` of them,
/// names, and `item` must be less than `items`.
#[inline(always)]
unsafe fn element_nodes<const N: usize>(
    connectivity: *const u32,
    coordinates: *const f64,
    item: usize,
    items: usize,
) -> [[f64; 3]; N] {
    let mut positions = [[0.0; 3]; N];
    positions.iter_mut().enumerate().for_each(|(a, position)| {
        // SAFETY: the node is one that the coordinates have.
        let node = unsafe { *connectivity.add(a * items + item) } as usize;
        position.iter_mut().enumerate().for_each(|(i, position)| {
            // SAFETY: the node has three coordinates.
            *position = unsafe { *coordinates.add(3 * node + i) }
        });
    });
    positions
}

/// Checks the element `item` of a block against `limit`, the largest squared angular frequency
/// that is stable, with the integration data of the element from `data`.
///
/// An element whose bound, plus `added`, is more than the limit or is not a number is recorded
/// in `unstable`, which takes the least of the [records](UnstableRecord) at the `step`. An
/// element whose bound is an error is recorded in `errors`, as [`block_forces`] records it.
///
/// # Safety
///
/// `connectivity` and `coordinates` must be those of [`element_nodes`], `masses` must be
/// `N * items` values, and `unstable` and `errors` must each be one aligned value.
#[inline(always)]
unsafe fn check_element<M, D, const G: usize, const N: usize>(
    data: &D,
    item: usize,
    items: usize,
    connectivity: *const u32,
    coordinates: *const f64,
    masses: *const f64,
    added: f64,
    limit: f64,
    step: u32,
    unstable: *mut u64,
    errors: *mut u64,
    model: &M,
) where
    M: StiffnessBound,
    D: IntegrationData<G, N>,
{
    // SAFETY: as for this function.
    let positions: ElementNodalCoordinates<N> =
        unsafe { element_nodes::<N>(connectivity, coordinates, item, items) }.into();
    let nodal_masses: ElementNodalLumpedMasses<N> = from_fn(|a| {
        // SAFETY: the element has a mass for each of its nodes.
        Mass::kilograms(unsafe { *masses.add(a * items + item) })
    })
    .into();
    match squared_frequency_bound_from(data, model, &ModelBound, &positions, &nodal_masses) {
        Ok(bound) => {
            let squared_frequency = bound.value() + added;
            if squared_frequency.is_nan() || squared_frequency > limit {
                // SAFETY: `unstable` is one aligned value, which the threads only take the least.
                unsafe { AtomicU64::from_ptr(unstable) }
                    .fetch_min(UnstableRecord::of(step, item).bits(), Ordering::Relaxed);
            }
        }
        Err(error) => {
            // SAFETY: `errors` is one aligned value, which the threads only take the least.
            unsafe { AtomicU64::from_ptr(errors) }
                .fetch_min(ErrorRecord::of(item, &error).bits(), Ordering::Relaxed);
        }
    }
}

/// The check of the time step of each of `items` elements of a block, one element to a thread,
/// with the gradient vectors and weights recomputed as [`block_forces`] does, and the elements
/// recorded as [`check_element`] says.
///
/// # Safety
///
/// The nodes, the tables and the `parameters` must be those of [`block_forces`], `masses` must
/// be `N * items` values, and `unstable` and `errors` must each be one aligned value.
#[inline(always)]
unsafe fn block_check<M, const G: usize, const N: usize>(
    items: u32,
    reference_coordinates: *const f64,
    derivatives: *const f64,
    weights: *const f64,
    connectivity: *const u32,
    coordinates: *const f64,
    masses: *const f64,
    limit: f64,
    step: u32,
    unstable: *mut u64,
    errors: *mut u64,
    parameters: *const f64,
    number_of_parameters: u32,
) where
    M: StiffnessBound + FromParameters,
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
        // SAFETY: the nodes are those of the reference coordinates.
        let nodes = unsafe { element_nodes::<N>(connectivity, reference_coordinates, item, items) };
        let element = Recomputed::new(nodes.into(), derivatives, weights);
        // SAFETY: as for this function.
        unsafe {
            check_element::<M, _, G, N>(
                &element,
                item,
                items,
                connectivity,
                coordinates,
                masses,
                0.0,
                limit,
                step,
                unstable,
                errors,
                &model,
            )
        }
    }
}

/// The check of the time step of each of `items` elements of a block of elements of one
/// integration point with hourglass control, one element to a thread, with what each has of its
/// own read from its `values` as [`block_forces_stabilized`] does, what it adds to the bound of
/// its point from `added`, and the elements recorded as [`check_element`] says.
///
/// # Safety
///
/// The nodes, the `values` and the `parameters` must be those of [`block_forces_stabilized`],
/// `masses` must be `8 * items` values, `added` must be `items` values, and `unstable` and
/// `errors` must each be one aligned value.
#[inline(always)]
unsafe fn block_check_stabilized<M>(
    items: u32,
    values: *const f64,
    connectivity: *const u32,
    coordinates: *const f64,
    masses: *const f64,
    added: *const f64,
    limit: f64,
    step: u32,
    unstable: *mut u64,
    errors: *mut u64,
    parameters: *const f64,
    number_of_parameters: u32,
) where
    M: StiffnessBound + FromParameters,
{
    const N: usize = 8;
    let item = thread();
    let items = items as usize;
    if item < items {
        // SAFETY: there are as many parameters as the model takes.
        let model = M::from_parameters(unsafe {
            from_raw_parts(parameters, number_of_parameters as usize)
        });
        let element = StabilizedElement::<N>::from_values(
            // SAFETY: the value is one of those that the element has.
            |k| unsafe { *values.add(k * items + item) },
        );
        // SAFETY: as for this function, and the element has what it adds.
        unsafe {
            check_element::<M, _, 1, N>(
                &element,
                item,
                items,
                connectivity,
                coordinates,
                masses,
                *added.add(item),
                limit,
                step,
                unstable,
                errors,
                &model,
            )
        }
    }
}

/// The second pass of the forces: the forces on `nodes` nodes, each the sum of the forces of the
/// elements that have it, in the order of the elements, added to the forces that it has.
///
/// A thread is a node. The contributions to the node `node` are
/// `contributions[offsets[node]..offsets[node + 1]]`, in the order of the elements, and each is
/// where the force of an element is in each of the three planes of components of `forces`, which
/// are `plane` values apart: the node `a` of an element `item` is `a * items + item`, as in
/// [`block_forces`]. The forces on the nodes are `nodal_forces[3 * node + i]`.
///
/// The sum is only additions, in the order of the elements, so the forces are those that the host
/// adds, whether the operations are fused or not.
///
/// # Safety
///
/// `offsets` must be `nodes + 1` values and `contributions` what they say, each within a plane,
/// `forces` must be three planes of `plane` values, and `nodal_forces` must be `3 * nodes`
/// values, at least.
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
        let (first, last) = unsafe { (*offsets.add(node), *offsets.add(node + 1)) };
        // SAFETY: the node has three forces.
        let mut sum = unsafe {
            [
                *nodal_forces.add(3 * node),
                *nodal_forces.add(3 * node + 1),
                *nodal_forces.add(3 * node + 2),
            ]
        };
        for k in first..last {
            // SAFETY: the contribution is one of those that the offsets give.
            let contribution = unsafe { *contributions.add(k as usize) } as usize;
            sum.iter_mut().enumerate().for_each(|(i, value)| {
                // SAFETY: the contribution is within a plane, and there are three of them.
                *value += unsafe { *forces.add(i * plane + contribution) }
            });
        }
        sum.iter().enumerate().for_each(|(i, &value)| {
            // SAFETY: the node has three forces.
            unsafe { *nodal_forces.add(3 * node + i) = value }
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
            /// # Safety
            ///
            /// As for [`block_check`].
            #[unsafe(export_name = concat!("block_check_", $name, "_g", $points, "_n", $nodes))]
            pub unsafe extern "ptx-kernel" fn check(
                items: u32,
                reference_coordinates: *const f64,
                derivatives: *const f64,
                weights: *const f64,
                connectivity: *const u32,
                coordinates: *const f64,
                masses: *const f64,
                limit: f64,
                step: u32,
                unstable: *mut u64,
                errors: *mut u64,
                parameters: *const f64,
                number_of_parameters: u32,
            ) {
                // SAFETY: as for the function that this calls.
                unsafe {
                    block_check::<$model, $points, $nodes>(
                        items,
                        reference_coordinates,
                        derivatives,
                        weights,
                        connectivity,
                        coordinates,
                        masses,
                        limit,
                        step,
                        unstable,
                        errors,
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

/// A kernel for the elements of one point with hourglass control of a block of a model, with
/// `$nodes` nodes, named `block_forces_<name>_stabilized_n<nodes>`.
macro_rules! stabilized {
    ($model:ty, $name:literal, $nodes:literal) => {
        const _: () = {
            assert!(
                same(<$model as FromParameters>::NAME, $name),
                "the name of the kernels is not the name of the model"
            );
            /// # Safety
            ///
            /// As for [`block_forces_stabilized`].
            #[unsafe(export_name = concat!("block_forces_", $name, "_stabilized_n", $nodes))]
            pub unsafe extern "ptx-kernel" fn kernel(
                items: u32,
                values: *const f64,
                connectivity: *const u32,
                coordinates: *const f64,
                forces: *mut f64,
                record: *mut u64,
                parameters: *const f64,
                number_of_parameters: u32,
            ) {
                // SAFETY: as for the function that this calls.
                unsafe {
                    block_forces_stabilized::<$model>(
                        items,
                        values,
                        connectivity,
                        coordinates,
                        forces,
                        record,
                        parameters,
                        number_of_parameters,
                    )
                }
            }
            /// # Safety
            ///
            /// As for [`block_check_stabilized`].
            #[unsafe(export_name = concat!("block_check_", $name, "_stabilized_n", $nodes))]
            pub unsafe extern "ptx-kernel" fn check(
                items: u32,
                values: *const f64,
                connectivity: *const u32,
                coordinates: *const f64,
                masses: *const f64,
                added: *const f64,
                limit: f64,
                step: u32,
                unstable: *mut u64,
                errors: *mut u64,
                parameters: *const f64,
                number_of_parameters: u32,
            ) {
                // SAFETY: as for the function that this calls.
                unsafe {
                    block_check_stabilized::<$model>(
                        items,
                        values,
                        connectivity,
                        coordinates,
                        masses,
                        added,
                        limit,
                        step,
                        unstable,
                        errors,
                        parameters,
                        number_of_parameters,
                    )
                }
            }
        };
    };
}

stabilized!(NeoHookean, "neo_hookean", 8);
stabilized!(SaintVenantKirchhoff, "saint_venant_kirchhoff", 8);

kernels! {
    models: [
        NeoHookean => "neo_hookean",
        SaintVenantKirchhoff => "saint_venant_kirchhoff",
    ],
    shapes: [(1, 4), (4, 4), (8, 8), (6, 6), (8, 5)],
}
