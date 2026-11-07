//! Explicit dynamics on a CUDA device.
//!
//! The nodal forces of a run are computed on the device in two passes, as on the threads: the
//! forces of the elements, and then the sum of those that each node has, in the order of the
//! elements. What does not change between steps lives on the device for the run.

mod cuda;
mod resident;
#[cfg(test)]
mod test;

pub(crate) use resident::Resident;

/// The degrees of freedom of a run that are moved, and where they are held at each step.
///
/// What they are held at is a function of time that only the host can evaluate, so it is
/// evaluated for a number of steps at a time and given to the device as a table.
pub(crate) struct Moved<'a> {
    /// The degrees of freedom, as `3 * node + component`.
    pub dofs: Vec<u32>,
    /// What they are held at after a number of steps, in the order of `dofs`.
    pub values: Box<dyn FnMut(usize) -> Vec<f64> + 'a>,
}

use crate::constitutive::solid::elastic::GpuModel;
use core::fmt::{self, Display, Formatter};

/// The floating-point type that a device computes with.
///
/// Many devices compute `f64` at a small fraction of the rate of `f32`, and the library
/// computes with `f64`, so that a device in `F32` agrees with it only to `f32`.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum GpuPrecision {
    /// Double precision.
    #[default]
    F64,
    /// Single precision.
    F32,
}

/// Where the device gets the gradient vectors of the shape functions of the elements.
///
/// Reading them costs memory traffic, and making them costs arithmetic, so which is faster
/// depends on the device and the precision: a device that is limited by its memory does better to
/// make them, and one that is limited by its arithmetic, as many are in `f64`, to read them.
///
/// On an H100 in `f64` making them is the faster for every element, by 20 to 25% for the
/// kernels of the library, so that is the default. A device whose `f64` is slow, as those of
/// many workstations and consumer cards are, may do better with [`Stored`](Self::Stored).
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum GpuGradients {
    /// Each element keeps its own, which are read at every step.
    Stored,
    /// They are made at every step from the reference coordinates of the nodes of the element,
    /// as an element makes them when it is made. Only elements of the three dimensions whose
    /// shape functions are those of their nodes can do this, and the block of any other keeps
    /// its own.
    #[default]
    OnTheFly,
}

/// Which kernels run the forces and the steps on the device.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum GpuKernels {
    /// Kernels written for the device in CUDA, which are compiled when the run starts, in either
    /// precision.
    #[default]
    Handwritten,
    /// The code of the library itself, built to PTX in advance, which the driver compiles when
    /// the run starts. It is double precision, with either way of getting the gradient vectors,
    /// and only for the models and the sizes of elements that it has kernels for: the linear
    /// hexahedron, tetrahedron, wedge and pyramid.
    Library,
}

/// How a device is used for a run.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct GpuOptions {
    /// The ordinal of the device, as CUDA numbers them.
    pub device: usize,
    /// The precision that the device computes with.
    pub precision: GpuPrecision,
    /// Where the device gets the gradient vectors.
    pub gradients: GpuGradients,
    /// Which kernels run on the device.
    pub kernels: GpuKernels,
    /// How many elements of a block the kernels of the library take at a time, or none for all of
    /// them.
    ///
    /// The forces of the elements of a block go to an array between the kernel that makes them
    /// and the one that sums them on the nodes, which is as large as the block and so is read
    /// from the memory of the device. In chunks of elements that are small enough for that array
    /// to stay in the cache of the device, which is tens of megabytes, it is not. Each node still
    /// gets the forces of its elements in the order of the elements, whatever the chunks are, so
    /// the results are the same to the bit. The kernels written for the device do not use this.
    pub chunk: usize,
}

/// What can go wrong on a device.
#[derive(Debug)]
pub enum GpuError {
    /// There is no CUDA, or no such device.
    Unavailable(String),
    /// The model or its elements have no kernels.
    Unsupported(String),
    /// The driver or the compiler reported an error.
    Cuda(String),
    /// An element of a block has a Jacobian that is not positive.
    Item {
        /// The block, in the order of the model.
        block: usize,
        /// The element of the block.
        item: usize,
    },
}

impl Display for GpuError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unavailable(message) => write!(f, "No CUDA device is available: {message}."),
            Self::Unsupported(message) => write!(f, "The device cannot compute this: {message}."),
            Self::Cuda(message) => write!(f, "CUDA: {message}"),
            Self::Item { block, item } => write!(
                f,
                "Element {item} of block {block} has a Jacobian that is not positive."
            ),
        }
    }
}

impl core::error::Error for GpuError {}

/// A block of elements as a device takes it: plain arrays, in the order of the elements.
#[derive(Clone)]
pub struct GpuItems {
    /// The constitutive model of the block.
    pub model: GpuModel,
    /// The nodes of each element.
    pub nodes_per_item: usize,
    /// The integration points of each element.
    pub points_per_item: usize,
    /// The nodes of the elements, `nodes_per_item` for each.
    pub connectivity: Vec<u32>,
    /// For each element, the gradient vectors of its nodes at each of its integration points
    /// (point, then node, then component), and then the weights of the points.
    pub reference: Vec<f64>,
    /// The offsets of the contributions to each node, from the index of the elements.
    pub offsets: Vec<u32>,
    /// The contributions to the nodes, as `nodes_per_item * element + position of the node`.
    /// These arrays are by element, and [`device_layout`](Self::device_layout) puts the
    /// element last for the kernels.
    pub contributions: Vec<u32>,
    /// The nodes that the offsets are for, if they are not all of the nodes in order, which is
    /// how a chunk of a block has them, as it only touches some.
    pub node_list: Vec<u32>,
    /// What makes the gradient vectors from the reference coordinates, if the elements can.
    pub shape: Option<ShapeTables>,
    /// The kernels of the library for the block, if it has them.
    pub library: Option<LibraryBlock>,
}

/// A block as the kernels of the library take it: with the same arrays as the kernels written for
/// it, and the name of the kernel that makes the forces of its elements.
#[derive(Clone)]
pub struct LibraryBlock {
    /// The name of the kernel that computes the forces of the elements.
    pub forces_kernel: String,
    /// The name of the kernel that does so with the gradient vectors made on the fly.
    pub on_the_fly_kernel: String,
}

impl LibraryBlock {
    /// The kernels of the library for a block of a model, in which each element has `nodes` nodes
    /// and `points` integration points, if the library has them. They are named for the model
    /// and the sizes, which are all that the code of the library asks of an element.
    pub fn new(model: &GpuModel, points: usize, nodes: usize) -> Option<Self> {
        let suffix = format!("{}_g{points}_n{nodes}", model.name());
        let block = Self {
            forces_kernel: format!("block_forces_{suffix}"),
            on_the_fly_kernel: format!("block_forces_on_the_fly_{suffix}"),
        };
        (resident::library_has(&block.forces_kernel)
            && resident::library_has(&block.on_the_fly_kernel))
        .then_some(block)
    }
}

/// The shape functions of the elements of a block, at the integration points of its parametric
/// element, from which gradient vectors are made on the fly.
#[derive(Clone)]
pub struct ShapeTables {
    /// The derivatives of the shape functions with respect to the parametric coordinates, as
    /// point, then node, then component.
    pub derivatives: Vec<f64>,
    /// The weights of the integration points of the parametric element.
    pub weights: Vec<f64>,
}

impl GpuItems {
    pub fn number_of_items(&self) -> usize {
        self.connectivity.len() / self.nodes_per_item
    }
    /// The block as chunks of at most `size` consecutive elements, each a block of its own that
    /// has the nodes that its elements touch, with the index of its first element in this block.
    ///
    /// The contributions to a node are in the order of the elements, so those of a chunk are the
    /// run of them that its elements make, and the sum of a node over the chunks, one after the
    /// other, is its sum over the elements.
    pub(crate) fn chunks(&self, size: usize) -> Vec<(usize, GpuItems)> {
        let items = self.number_of_items();
        if items == 0 || size == 0 || size >= items {
            return vec![(0, self.whole())];
        }
        let width = self.nodes_per_item;
        let count = items.div_ceil(size);
        // The nodes, offsets and contributions of each chunk, as they are built.
        let mut built = (0..count)
            .map(|_| (Vec::<u32>::new(), vec![0_u32], Vec::<u32>::new()))
            .collect::<Vec<_>>();
        let node_of = |position: usize| {
            if self.node_list.is_empty() {
                position as u32
            } else {
                self.node_list[position]
            }
        };
        for position in 0..self.offsets.len() - 1 {
            let (start, end) = (
                self.offsets[position] as usize,
                self.offsets[position + 1] as usize,
            );
            let chunk_of = |k: usize| self.contributions[k] as usize / width / size;
            let mut k = start;
            while k < end {
                let (chunk, from) = (chunk_of(k), k);
                while k < end && chunk_of(k) == chunk {
                    k += 1;
                }
                let (nodes, offsets, contributions) = &mut built[chunk];
                nodes.push(node_of(position));
                contributions.extend(
                    self.contributions[from..k]
                        .iter()
                        .map(|&contribution| contribution - (chunk * size * width) as u32),
                );
                offsets.push(contributions.len() as u32);
            }
        }
        let reference_size = self.reference.len() / items;
        built
            .into_iter()
            .enumerate()
            .map(|(chunk, (node_list, offsets, contributions))| {
                let (first, last) = (chunk * size, ((chunk + 1) * size).min(items));
                (
                    first,
                    GpuItems {
                        model: self.model,
                        nodes_per_item: width,
                        points_per_item: self.points_per_item,
                        connectivity: self.connectivity[first * width..last * width].to_vec(),
                        reference: self.reference[first * reference_size..last * reference_size]
                            .to_vec(),
                        offsets,
                        contributions,
                        node_list,
                        shape: self.shape.clone(),
                        library: self.library.clone(),
                    },
                )
            })
            .collect()
    }
    /// The block as one chunk of itself, which is the nodes that it touches.
    fn whole(&self) -> GpuItems {
        let mut whole = self.clone();
        if whole.node_list.is_empty() {
            whole.node_list = (0..(whole.offsets.len().max(1) - 1) as u32).collect();
        }
        whole
    }
    /// The arrays as the kernels read them, with the element last, so that the threads of a
    /// warp, which are consecutive elements, read consecutive addresses.
    pub(crate) fn device_layout(&self) -> DeviceLayout {
        let items = self.number_of_items();
        let reference_size = self.reference.len() / items.max(1);
        DeviceLayout {
            connectivity: transpose(&self.connectivity, self.nodes_per_item, items),
            reference: transpose(&self.reference, reference_size, items),
            // The position of the node in the element, then the element, instead of the
            // element and then the position.
            contributions: self
                .contributions
                .iter()
                .map(|&contribution| {
                    let (item, local) = (
                        contribution as usize / self.nodes_per_item,
                        contribution as usize % self.nodes_per_item,
                    );
                    (local * items + item) as u32
                })
                .collect(),
        }
    }
}

/// The arrays of a block as the kernels read them.
pub(crate) struct DeviceLayout {
    /// The node of each element, as `position * items + item`.
    pub connectivity: Vec<u32>,
    /// The reference values of each element, as `value * items + item`.
    pub reference: Vec<f64>,
    /// The contributions to the nodes, as `position * items + item`.
    pub contributions: Vec<u32>,
}

/// `values`, which are `width` for each of `items`, with the item last instead of first.
fn transpose<T: Copy + Default>(values: &[T], width: usize, items: usize) -> Vec<T> {
    let mut transposed = vec![T::default(); values.len()];
    values
        .chunks_exact(width.max(1))
        .enumerate()
        .for_each(|(item, row)| {
            row.iter()
                .enumerate()
                .for_each(|(value, &x)| transposed[value * items + item] = x)
        });
    transposed
}
