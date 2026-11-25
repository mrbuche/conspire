mod dynamics;

pub use dynamics::ExplicitRun;

use self::dynamics::Kernels;
use crate::{
    domain::{
        ElementModelError, Model, NodalCoordinates,
        block::device::{DeviceBlock, DeviceBlocks, DeviceKind},
        solid::{NodalForcesSolid, elastic::ElasticElements},
    },
    fem::element::ErrorRecord,
    gpu::{
        Application, Buffer, Device, Function, GpuError, Module, Resources,
        kernels::{block_forces, block_forces_stabilized, ptx},
    },
    math::Tensor,
};
use core::{
    error::Error,
    fmt::{self, Debug, Display, Formatter},
};

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

impl Error for DeviceError {}

/// A block of a model on the device.
struct BlockOnDevice {
    kind: DeviceKind,
    forces_name: String,
    forces_kernel: Function,
    gather_kernel: Function,
    items: u32,
    plane: u32,
    nodes: u32,
    number_of_parameters: u32,
    derivatives: Buffer<f64>,
    weights: Buffer<f64>,
    values: Buffer<f64>,
    connectivity: Buffer<u32>,
    parameters: Buffer<f64>,
    offsets: Buffer<u32>,
    contributions: Buffer<u32>,
    forces: Buffer<f64>,
    record: Buffer<u64>,
}

impl BlockOnDevice {
    fn new(device: &Device, module: &Module, block: &DeviceBlock) -> Result<Self, GpuError> {
        let layout = &block.layout;
        let name = match block.kind {
            DeviceKind::Full => block_forces(block.model, block.points, layout.nodes_per_item),
            DeviceKind::Stabilized => block_forces_stabilized(block.model, layout.nodes_per_item),
        };
        let index = |value: usize| {
            u32::try_from(value)
                .map_err(|_| GpuError::Cuda("a block is too large for a device".to_string()))
        };
        Ok(Self {
            kind: block.kind,
            forces_kernel: module
                .function(&name)
                .map_err(|_| GpuError::NoKernel(name.clone()))?,
            forces_name: name,
            gather_kernel: module.function("gather")?,
            items: index(layout.items)?,
            plane: index(layout.plane())?,
            nodes: index(layout.number_of_nodes())?,
            number_of_parameters: index(block.parameters.len())?,
            derivatives: device.buffer_of(&block.derivatives)?,
            weights: device.buffer_of(&block.weights)?,
            values: device.buffer_of(&block.values)?,
            connectivity: device.buffer_of(&layout.connectivity)?,
            parameters: device.buffer_of(&block.parameters)?,
            offsets: device.buffer_of(&layout.offsets)?,
            contributions: device.buffer_of(&layout.contributions)?,
            forces: device.buffer(3 * layout.plane())?,
            record: device.buffer(1)?,
        })
    }
}

/// A model on a device, which has what does not change between calls: the reference coordinates,
/// and for each block the connectivity, the parameters of the model, what the elements share, and
/// the layout of the second pass of the forces. A call gives it the coordinates, and gets the
/// nodal forces.
pub struct Resident {
    device: Device,
    nodes: usize,
    reference_coordinates: Buffer<f64>,
    coordinates: Buffer<f64>,
    nodal_forces: Buffer<f64>,
    blocks: Vec<BlockOnDevice>,
    fused: bool,
    /// The kernels of the steps of dynamics, which are loaded when a run is first started.
    kernels: Option<Kernels>,
    _module: Module,
}

impl Resident {
    /// Puts a model on the device of the given ordinal, with the operations of the kernels fused
    /// if `fused`, which is faster and differs from the host in the last bits.
    pub fn new<B>(ordinal: usize, model: &Model<B, 3>, fused: bool) -> Result<Self, GpuError>
    where
        B: DeviceBlocks,
    {
        let device = Device::new(ordinal)?;
        let module = device.module_from_ptx(&ptx(Application::Elements, fused)?)?;
        let reference = model.coordinates.as_flat();
        let mut described = Vec::new();
        model.device_blocks(&mut described);
        let blocks = described
            .iter()
            .map(|block| BlockOnDevice::new(&device, &module, block))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            nodes: reference.len() / 3,
            reference_coordinates: device.buffer_of(reference)?,
            coordinates: device.buffer(reference.len())?,
            nodal_forces: device.buffer(reference.len())?,
            blocks,
            fused,
            kernels: None,
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
    /// Gives the device the coordinates of the nodes, from the nodes themselves.
    pub fn set_coordinates(&self, coordinates: &NodalCoordinates<3>) -> Result<(), GpuError> {
        let coordinates = coordinates.as_flat();
        assert_eq!(
            coordinates.len(),
            3 * self.nodes,
            "a model is run with the nodes that it has"
        );
        self.coordinates.upload(coordinates)
    }
    /// Computes the nodal forces for the coordinates that the device has, and waits for them.
    pub fn run(&self) -> Result<(), GpuError> {
        self.forget_errors()?;
        self.launch_forces()?;
        self.device.synchronize()
    }
    /// Clears the errors that the elements have found.
    fn forget_errors(&self) -> Result<(), GpuError> {
        self.blocks
            .iter()
            .try_for_each(|block| block.record.fill(u32::MAX))
    }
    /// Starts the computation of the nodal forces for the coordinates that the device has, and
    /// does not wait for it, so that what follows it on the device is in order after it.
    fn launch_forces(&self) -> Result<(), GpuError> {
        self.nodal_forces.fill(0)?;
        for block in &self.blocks {
            match block.kind {
                DeviceKind::Full => self.device.launch(
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
                        &block.record,
                        &block.parameters,
                        &block.number_of_parameters,
                    ],
                )?,
                DeviceKind::Stabilized => self.device.launch(
                    &block.forces_kernel,
                    block.items as usize,
                    THREADS_PER_BLOCK,
                    &[
                        &block.items,
                        &block.values,
                        &block.connectivity,
                        &self.coordinates,
                        &block.forces,
                        &block.record,
                        &block.parameters,
                        &block.number_of_parameters,
                    ],
                )?,
            }
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
        Ok(())
    }
    /// What the kernels of each block need of a multiprocessor of the device, for the blocks of
    /// threads that they are launched in: the kernel of the forces of the elements, and then the
    /// second pass.
    pub fn resources(&self) -> Result<Vec<(String, Resources)>, GpuError> {
        let mut resources = Vec::new();
        for block in &self.blocks {
            resources.push((
                block.forces_name.clone(),
                block.forces_kernel.resources(THREADS_PER_BLOCK)?,
            ));
            resources.push((
                "gather".to_string(),
                block.gather_kernel.resources(THREADS_PER_BLOCK)?,
            ));
        }
        Ok(resources)
    }
    /// The first error that the last computation found, with the block that it is in, which is
    /// the first block that has one, and in it the first element, since the blocks add in order.
    pub fn first_error(&self) -> Result<Option<(usize, ErrorRecord)>, GpuError> {
        for (index, block) in self.blocks.iter().enumerate() {
            let mut word = [u64::MAX];
            block.record.download(&mut word)?;
            let record = ErrorRecord::from_bits(word[0]);
            if record.is_error() {
                return Ok(Some((index, record)));
            }
        }
        Ok(None)
    }
    /// The nodal forces that the last computation made, in new nodes.
    ///
    /// A call that makes them again and again is better to keep nodes and put the forces into
    /// them with [`download_into`](Self::download_into): copying into nodes that are made for it
    /// has been measured to take several times as long as copying into nodes that are kept.
    pub fn download(&self) -> Result<NodalForcesSolid<3>, GpuError> {
        let mut forces = NodalForcesSolid::<3>::zero(self.nodes);
        self.download_into(&mut forces)?;
        Ok(forces)
    }
    /// The nodal forces that the last computation made, put into `forces`, which have the nodes
    /// of the model, and whatever they held does not matter.
    pub fn download_into(&self, forces: &mut NodalForcesSolid<3>) -> Result<(), GpuError> {
        assert_eq!(
            forces.len(),
            self.nodes,
            "a model is run with the nodes that it has"
        );
        self.nodal_forces.download(forces.as_flat_mut())
    }
    /// The nodal forces of the model for the coordinates, from the device, in new nodes.
    ///
    /// The model must be the one that this was made of. If an element has an error on the device,
    /// the host computes the model, and reports the error that it finds, which is that of the
    /// first element that has one, as the blocks and the elements are in the same order.
    pub fn nodal_forces<B>(
        &self,
        model: &Model<B, 3>,
        coordinates: &NodalCoordinates<3>,
    ) -> Result<NodalForcesSolid<3>, DeviceError>
    where
        B: ElasticElements<3>,
    {
        let mut forces = NodalForcesSolid::zero(self.nodes);
        self.nodal_forces_into(model, coordinates, &mut forces)?;
        Ok(forces)
    }
    /// The nodal forces of the model for the coordinates, from the device, put into `forces`, as
    /// [`nodal_forces`](Self::nodal_forces) makes them, and into nodes that are kept by a run that
    /// takes them many times.
    pub fn nodal_forces_into<B>(
        &self,
        model: &Model<B, 3>,
        coordinates: &NodalCoordinates<3>,
        forces: &mut NodalForcesSolid<3>,
    ) -> Result<(), DeviceError>
    where
        B: ElasticElements<3>,
    {
        self.compute(coordinates)?;
        if let Some((block, record)) = self.first_error()? {
            return Err(match model.nodal_forces(coordinates) {
                Err(error) => DeviceError::Element(error),
                Ok(_) => DeviceError::Gpu(GpuError::Disagreement {
                    block,
                    item: record.item(),
                    point: record.point(),
                }),
            });
        }
        Ok(self.download_into(forces)?)
    }
}
