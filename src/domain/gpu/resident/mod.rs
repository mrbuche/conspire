//! The forces of a model on a device.
//!
//! What does not change between calls lives on the device: the reference coordinates, and for
//! each block the connectivity, the parameters of the model, what the elements share, and the
//! second pass of the forces. A call gives the device the coordinates, and gets the nodal forces.

use super::{
    DeviceBlock, DeviceBlocks, GpuError,
    cuda::{Buffer, Device, Function, Module},
    kernels,
};
use crate::{
    domain::{ElementModelError, Model, NodalCoordinates},
    fem::{
        element::Failure,
        solid::{NodalForcesSolid, elastic::ElasticElements},
    },
    math::{Tensor, TensorRank1Vec},
    units::Length,
};
use core::fmt::{self, Debug, Display, Formatter};

const THREADS_PER_BLOCK: usize = 128;

/// Possible errors encountered in the forces on a device.
pub enum DeviceError {
    /// The device or its kernels.
    Gpu(GpuError),
    /// An element of the model, as the host reports it.
    Element(ElementModelError),
}

impl From<GpuError> for DeviceError {
    fn from(error: GpuError) -> Self {
        Self::Gpu(error)
    }
}

impl Display for DeviceError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Gpu(error) => write!(f, "{error}"),
            Self::Element(error) => write!(f, "{error}"),
        }
    }
}

impl Debug for DeviceError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "\n{self}\n")
    }
}

impl core::error::Error for DeviceError {}

/// A block of a model on the device.
struct BlockOnDevice {
    forces_kernel: Function,
    gather_kernel: Function,
    items: u32,
    plane: u32,
    nodes: u32,
    number_of_parameters: u32,
    derivatives: Buffer<f64>,
    weights: Buffer<f64>,
    connectivity: Buffer<u32>,
    parameters: Buffer<f64>,
    offsets: Buffer<u32>,
    contributions: Buffer<u32>,
    forces: Buffer<f64>,
    failure: Buffer<u64>,
}

impl BlockOnDevice {
    fn new(device: &Device, module: &Module, block: &DeviceBlock) -> Result<Self, GpuError> {
        let layout = &block.layout;
        let name = kernels::block_forces(block.model, block.points, layout.nodes_per_item);
        let index = |value: usize| {
            u32::try_from(value)
                .map_err(|_| GpuError::Cuda("a block is too large for a device".to_string()))
        };
        Ok(Self {
            forces_kernel: module
                .function(&name)
                .map_err(|_| GpuError::NoKernel(name))?,
            gather_kernel: module.function("gather")?,
            items: index(layout.items)?,
            plane: index(layout.plane())?,
            nodes: index(layout.number_of_nodes())?,
            number_of_parameters: index(block.parameters.len())?,
            derivatives: device.buffer_of(&block.derivatives)?,
            weights: device.buffer_of(&block.weights)?,
            connectivity: device.buffer_of(&layout.connectivity)?,
            parameters: device.buffer_of(&block.parameters)?,
            offsets: device.buffer_of(&layout.offsets)?,
            contributions: device.buffer_of(&layout.contributions)?,
            forces: device.buffer(3 * layout.plane())?,
            failure: device.buffer(1)?,
        })
    }
}

/// A model on a device: the nodal forces of its blocks.
pub struct Resident {
    device: Device,
    nodes: usize,
    reference_coordinates: Buffer<f64>,
    coordinates: Buffer<f64>,
    nodal_forces: Buffer<f64>,
    blocks: Vec<BlockOnDevice>,
    _module: Module,
}

impl Resident {
    /// Puts a model on the device of the given ordinal, with the operations of the kernels fused
    /// if `fused`, which is faster and differs from the host in the last bits (see
    /// [`ptx`](super::kernels::ptx)).
    pub fn new<B>(ordinal: usize, model: &Model<B, 3>, fused: bool) -> Result<Self, GpuError>
    where
        B: DeviceBlocks,
    {
        let device = Device::new(ordinal)?;
        let module = device.module_from_ptx(&kernels::ptx(fused)?)?;
        let reference = flatten(&model.coordinates);
        let mut described = Vec::new();
        model.device_blocks(&mut described);
        let blocks = described
            .iter()
            .map(|block| BlockOnDevice::new(&device, &module, block))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            nodes: reference.len() / 3,
            reference_coordinates: device.buffer_of(&reference)?,
            coordinates: device.buffer(reference.len())?,
            nodal_forces: device.buffer(reference.len())?,
            blocks,
            _module: module,
            device,
        })
    }
    /// The name of the device.
    pub fn device_name(&self) -> &str {
        self.device.name()
    }
    /// Computes the nodal forces for the coordinates on the device, and waits for them.
    pub fn compute(&self, coordinates: &NodalCoordinates<3>) -> Result<(), GpuError> {
        self.set_coordinates(coordinates)?;
        self.run()
    }
    /// Gives the device the coordinates of the nodes.
    pub fn set_coordinates(&self, coordinates: &NodalCoordinates<3>) -> Result<(), GpuError> {
        let coordinates = flatten(coordinates);
        assert_eq!(
            coordinates.len(),
            3 * self.nodes,
            "a model is run with the nodes that it has"
        );
        self.coordinates.upload(&coordinates)
    }
    /// Computes the nodal forces for the coordinates that the device has, and waits for them.
    pub fn run(&self) -> Result<(), GpuError> {
        self.nodal_forces.fill(0)?;
        for block in &self.blocks {
            block.failure.fill(u32::MAX)?;
            self.device.launch(
                &block.forces_kernel,
                block.items as usize,
                THREADS_PER_BLOCK,
                &[
                    &block.items,
                    &self.reference_coordinates,
                    &block.derivatives,
                    &block.weights,
                    &block.connectivity,
                    &self.coordinates,
                    &block.forces,
                    &block.failure,
                    &block.parameters,
                    &block.number_of_parameters,
                ],
            )?;
            self.device.launch(
                &block.gather_kernel,
                block.nodes as usize,
                THREADS_PER_BLOCK,
                &[
                    &block.nodes,
                    &block.plane,
                    &block.offsets,
                    &block.contributions,
                    &block.forces,
                    &self.nodal_forces,
                ],
            )?;
        }
        self.device.synchronize()
    }
    /// The first failure that the last computation found, with the block that it is in, which is
    /// the first block with one, and in it the first element, since the blocks add in order.
    pub fn failure(&self) -> Result<Option<(usize, Failure)>, GpuError> {
        for (index, block) in self.blocks.iter().enumerate() {
            let mut word = [u64::MAX];
            block.failure.download(&mut word)?;
            let failure = Failure::from_bits(word[0]);
            if failure.is_failure() {
                return Ok(Some((index, failure)));
            }
        }
        Ok(None)
    }
    /// The nodal forces that the last computation made.
    pub fn download(&self) -> Result<NodalForcesSolid<3>, GpuError> {
        let mut values = vec![0.0; 3 * self.nodes];
        self.nodal_forces.download(&mut values)?;
        Ok(values.as_chunks::<3>().0.to_vec().into())
    }
    /// The nodal forces of the model for the coordinates, from the device.
    ///
    /// The model must be the one that this was made of. If an element fails on the device, the
    /// host computes the model, and reports the error that it finds, which is that of the first
    /// element that fails, as the blocks and the elements are in the same order.
    pub fn nodal_forces<B>(
        &self,
        model: &Model<B, 3>,
        coordinates: &NodalCoordinates<3>,
    ) -> Result<NodalForcesSolid<3>, DeviceError>
    where
        B: ElasticElements<3>,
    {
        self.compute(coordinates)?;
        if let Some((block, failure)) = self.failure()? {
            return Err(match model.nodal_forces(coordinates) {
                Err(error) => DeviceError::Element(error),
                Ok(_) => DeviceError::Gpu(GpuError::Disagreement {
                    block,
                    item: failure.item(),
                    point: failure.point(),
                }),
            });
        }
        Ok(self.download()?)
    }
}

fn flatten<I>(coordinates: &TensorRank1Vec<3, I, Length>) -> Vec<f64> {
    coordinates
        .iter()
        .flat_map(|coordinate| coordinate.iter().map(|component| component.value()))
        .collect()
}
