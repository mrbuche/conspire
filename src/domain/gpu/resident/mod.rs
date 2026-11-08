//! The forces of a model on a device, and steps of dynamics with them.
//!
//! What does not change between calls lives on the device: the reference coordinates, and for
//! each block the connectivity, the parameters of the model, what the elements share, and the
//! second pass of the forces. A call gives the device the coordinates, and gets the nodal forces.
//!
//! A run of dynamics keeps its velocities and accelerations there as well, and takes steps of
//! velocity Verlet without anything going between the device and the host.

use super::{
    DeviceBlock, DeviceBlocks, GpuError,
    cuda::{Buffer, Device, Function, Module, Resources},
    kernels,
};
use crate::{
    domain::{
        ElementModelError, Model, NodalAccelerations, NodalCoordinates, NodalVelocities,
        solid::mass::NodalLumpedMasses,
    },
    fem::{
        element::Failure,
        solid::{NodalForcesSolid, elastic::ElasticElements},
    },
    math::{Tensor, TensorRank1Vec},
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
    forces_name: String,
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
                .map_err(|_| GpuError::NoKernel(name.clone()))?,
            forces_name: name,
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

/// The state of a run of dynamics on the device, and what steps it.
struct Dynamics {
    velocities: Buffer<f64>,
    accelerations: Buffer<f64>,
    masses: Buffer<f64>,
    held: Buffer<u32>,
}

/// A model on a device: the nodal forces of its blocks, and the steps of dynamics.
pub struct Resident {
    device: Device,
    nodes: usize,
    reference_coordinates: Buffer<f64>,
    coordinates: Buffer<f64>,
    nodal_forces: Buffer<f64>,
    blocks: Vec<BlockOnDevice>,
    kick_drift_kernel: Function,
    accelerate_kernel: Function,
    dynamics: Option<Dynamics>,
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
            kick_drift_kernel: module.function("kick_drift")?,
            accelerate_kernel: module.function("accelerate")?,
            dynamics: None,
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
        self.forget_failures()?;
        self.launch_forces()?;
        self.device.synchronize()
    }
    /// Clears the failures that the elements have found.
    fn forget_failures(&self) -> Result<(), GpuError> {
        self.blocks
            .iter()
            .try_for_each(|block| block.failure.fill(u32::MAX))
    }
    /// Starts the computation of the nodal forces for the coordinates that the device has, and
    /// does not wait for it, so that what follows it on the device is in order after it.
    fn launch_forces(&self) -> Result<(), GpuError> {
        self.nodal_forces.fill(0)?;
        for block in &self.blocks {
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
        Ok(())
    }
    /// Starts a run of dynamics from the coordinates and velocities, with the masses of the
    /// nodes, and the degrees of freedom `held`, which are indexed as `3 * node + component`, and
    /// have no velocity and no acceleration, as in dynamics on the host. The accelerations are
    /// those of the forces on the coordinates.
    pub fn start_dynamics(
        &mut self,
        masses: &NodalLumpedMasses,
        held: &[usize],
        coordinates: &NodalCoordinates<3>,
        velocities: &NodalVelocities<3>,
    ) -> Result<(), GpuError> {
        assert_eq!(
            masses.len(),
            self.nodes,
            "a model is run with the nodes that it has"
        );
        assert_eq!(
            velocities.len(),
            self.nodes,
            "a model is run with the nodes that it has"
        );
        let mut velocities = flatten(velocities);
        let mut flags = vec![0_u32; 3 * self.nodes];
        for &dof in held {
            flags[dof] = 1;
            velocities[dof] = 0.0;
        }
        self.set_coordinates(coordinates)?;
        let masses = masses.iter().map(|mass| mass.value()).collect::<Vec<_>>();
        self.dynamics = Some(Dynamics {
            velocities: self.device.buffer_of(&velocities)?,
            accelerations: self.device.buffer(3 * self.nodes)?,
            masses: self.device.buffer_of(&masses)?,
            held: self.device.buffer_of(&flags)?,
        });
        self.forget_failures()?;
        self.launch_forces()?;
        self.launch_accelerate(0.0, false)?;
        self.device.synchronize()?;
        self.failed()
    }
    /// Takes `steps` steps of velocity Verlet of size `dt`, on the device and without copying
    /// anything between the device and the host, and waits for them.
    ///
    /// An element that fails is not noticed until the steps are done, and then it is an error,
    /// and the state is of no use.
    pub fn steps(&self, dt: f64, steps: usize) -> Result<(), GpuError> {
        let dynamics = self.started()?;
        let count = u32::try_from(3 * self.nodes)
            .map_err(|_| GpuError::Cuda("a model is too large for a device".to_string()))?;
        self.forget_failures()?;
        for _ in 0..steps {
            self.device.launch(
                &self.kick_drift_kernel,
                3 * self.nodes,
                THREADS_PER_BLOCK,
                &[
                    &count,
                    &self.coordinates,
                    &dynamics.velocities,
                    &dynamics.accelerations,
                    &dt,
                ],
            )?;
            self.launch_forces()?;
            self.launch_accelerate(dt, true)?;
        }
        self.device.synchronize()?;
        self.failed()
    }
    fn started(&self) -> Result<&Dynamics, GpuError> {
        self.dynamics.as_ref().ok_or_else(|| {
            GpuError::Cuda("a run of dynamics is started before it is stepped".to_string())
        })
    }
    /// Starts the accelerations that balance the forces, and the velocities another half step on
    /// if `kick`.
    fn launch_accelerate(&self, dt: f64, kick: bool) -> Result<(), GpuError> {
        let dynamics = self.started()?;
        let count = u32::try_from(3 * self.nodes)
            .map_err(|_| GpuError::Cuda("a model is too large for a device".to_string()))?;
        self.device.launch(
            &self.accelerate_kernel,
            3 * self.nodes,
            THREADS_PER_BLOCK,
            &[
                &count,
                &self.nodal_forces,
                &dynamics.masses,
                &dynamics.held,
                &dynamics.accelerations,
                &dynamics.velocities,
                &dt,
                &u32::from(kick),
            ],
        )
    }
    /// An error if an element failed in the computations since the failures were cleared.
    fn failed(&self) -> Result<(), GpuError> {
        match self.failure()? {
            Some((block, failure)) => Err(GpuError::Failed {
                block,
                item: failure.item(),
                point: failure.point(),
            }),
            None => Ok(()),
        }
    }
    /// The coordinates, velocities and accelerations of the run of dynamics.
    pub fn dynamics_state(
        &self,
    ) -> Result<
        (
            NodalCoordinates<3>,
            NodalVelocities<3>,
            NodalAccelerations<3>,
        ),
        GpuError,
    > {
        let dynamics = self.started()?;
        let download = |buffer: &Buffer<f64>| -> Result<Vec<[f64; 3]>, GpuError> {
            let mut values = vec![0.0; 3 * self.nodes];
            buffer.download(&mut values)?;
            Ok(values.as_chunks::<3>().0.to_vec())
        };
        Ok((
            download(&self.coordinates)?.into(),
            download(&dynamics.velocities)?.into(),
            download(&dynamics.accelerations)?.into(),
        ))
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

fn flatten<I, U>(components: &TensorRank1Vec<3, I, U>) -> Vec<f64> {
    components
        .iter()
        .flat_map(|node| node.iter().map(|component| component.value()))
        .collect()
}
