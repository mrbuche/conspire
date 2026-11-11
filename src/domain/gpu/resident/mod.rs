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
    math::Tensor,
};
use core::{
    cell::Cell,
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

/// An external force on a run of dynamics, which is subtracted from the forces of the elements.
pub enum ImposedLoad {
    /// A force on each degree of freedom, as `3 * node + component`, which does not change.
    Constant(Vec<f64>),
    /// The weight of each node, whose force is the weight times three components that are a
    /// function of the step, which the [`Schedule`] gives.
    Weighted(Vec<f64>),
}

/// What is imposed on a run of dynamics, which does not change from step to step.
///
/// The loads are subtracted from the forces in the order that they are given, as the host
/// subtracts them, because the order is part of the last bits.
#[derive(Default)]
pub struct Imposed {
    /// The degrees of freedom, as `3 * node + component`, that are held where they are: they have
    /// no acceleration.
    pub fixed: Vec<usize>,
    /// The degrees of freedom whose coordinates are set at every step, which have no acceleration
    /// either.
    pub moved: Vec<usize>,
    /// The external forces.
    pub loads: Vec<ImposedLoad>,
}

/// What is imposed on a run of dynamics as a function of the step, which is the row of a table:
/// row 0 is the start, and row `r` is the state after `r` steps.
///
/// A device is given as many rows as it takes steps at a time, not those of the whole run.
#[derive(Default)]
pub struct Schedule<'a> {
    /// What each of the moved degrees of freedom is set to, in the order of [`Imposed::moved`].
    pub targets: Option<&'a dyn Fn(usize) -> Vec<f64>>,
    /// The components of each weighted load, in the order of the loads.
    pub components: Vec<&'a dyn Fn(usize) -> [f64; 3]>,
}

const TABLE_VALUES: usize = 1 << 22;
const MAX_ROWS: usize = 1024;

struct Moved {
    count: u32,
    dofs: Buffer<u32>,
    table: Buffer<f64>,
}

enum LoadOnDevice {
    Constant(Buffer<f64>),
    Weighted {
        weights: Buffer<f64>,
        table: Buffer<f64>,
    },
}

/// The state of a run of dynamics on the device, and what is imposed on it.
struct Dynamics {
    velocities: Buffer<f64>,
    accelerations: Buffer<f64>,
    masses: Buffer<f64>,
    held: Buffer<u32>,
    moved: Option<Moved>,
    loads: Vec<LoadOnDevice>,
    /// The rows that the tables hold.
    rows: usize,
    /// The row of the state of the run.
    row: Cell<usize>,
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
    set_targets_kernel: Function,
    subtract_constant_kernel: Function,
    subtract_weighted_kernel: Function,
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
        let reference = model.coordinates.components();
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
            kick_drift_kernel: module.function("kick_drift")?,
            accelerate_kernel: module.function("accelerate")?,
            set_targets_kernel: module.function("set_targets")?,
            subtract_constant_kernel: module.function("subtract_constant")?,
            subtract_weighted_kernel: module.function("subtract_weighted")?,
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
        assert_eq!(
            coordinates.len(),
            self.nodes,
            "a model is run with the nodes that it has"
        );
        // From the memory of the nodes, which is not copied first.
        self.coordinates.upload(coordinates.components())
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
    /// nodes, and what is imposed on it; `schedule` says what is imposed at the start.
    ///
    /// The velocities are those given: dynamics on the host has the fixed degrees of freedom
    /// without velocity, and the caller makes them so. The accelerations at the start are those of
    /// the forces on the coordinates, less the loads.
    pub fn start_dynamics(
        &mut self,
        masses: &NodalLumpedMasses,
        coordinates: &NodalCoordinates<3>,
        velocities: &NodalVelocities<3>,
        imposed: &Imposed,
        schedule: &Schedule<'_>,
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
        let dofs = 3 * self.nodes;
        let index = |dof: usize| {
            u32::try_from(dof)
                .ok()
                .filter(|_| dof < dofs)
                .ok_or_else(|| {
                    GpuError::Invalid(format!("degree of freedom {dof} is not in the model"))
                })
        };
        let mut flags = vec![0_u32; dofs];
        for &dof in imposed.fixed.iter().chain(&imposed.moved) {
            flags[index(dof)? as usize] = 1;
        }
        let moved = if imposed.moved.is_empty() {
            None
        } else {
            let list = imposed
                .moved
                .iter()
                .map(|&dof| index(dof))
                .collect::<Result<Vec<_>, _>>()?;
            Some((list.len(), list))
        };
        let rows = (TABLE_VALUES / moved.as_ref().map_or(3, |(count, _)| (*count).max(3)))
            .clamp(1, MAX_ROWS);
        let moved = moved
            .map(|(count, list)| -> Result<Moved, GpuError> {
                Ok(Moved {
                    count: count as u32,
                    dofs: self.device.buffer_of(&list)?,
                    table: self.device.buffer(count * rows)?,
                })
            })
            .transpose()?;
        let loads = imposed
            .loads
            .iter()
            .map(|load| -> Result<LoadOnDevice, GpuError> {
                Ok(match load {
                    ImposedLoad::Constant(forces) if forces.len() == dofs => {
                        LoadOnDevice::Constant(self.device.buffer_of(forces)?)
                    }
                    ImposedLoad::Constant(forces) => {
                        return Err(GpuError::Invalid(format!(
                            "a load has {} values, and the model has {dofs} degrees of freedom",
                            forces.len()
                        )));
                    }
                    ImposedLoad::Weighted(weights) if weights.len() == self.nodes => {
                        LoadOnDevice::Weighted {
                            weights: self.device.buffer_of(weights)?,
                            table: self.device.buffer(3 * rows)?,
                        }
                    }
                    ImposedLoad::Weighted(weights) => {
                        return Err(GpuError::Invalid(format!(
                            "a load has {} weights, and the model has {} nodes",
                            weights.len(),
                            self.nodes
                        )));
                    }
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        self.set_coordinates(coordinates)?;
        self.dynamics = Some(Dynamics {
            velocities: self.device.buffer_of(velocities.components())?,
            accelerations: self.device.buffer(dofs)?,
            masses: self
                .device
                .buffer_of(&masses.iter().map(|mass| mass.value()).collect::<Vec<_>>())?,
            held: self.device.buffer_of(&flags)?,
            moved,
            loads,
            rows,
            row: Cell::new(0),
        });
        let dynamics = self.started()?;
        self.check(dynamics, schedule)?;
        self.upload_tables(dynamics, schedule, 0, 1)?;
        self.forget_failures()?;
        self.launch_targets(dynamics, 0)?;
        self.launch_forces()?;
        self.launch_loads(dynamics, 0)?;
        self.launch_accelerate(0.0, false)?;
        self.device.synchronize()?;
        self.failed()
    }
    /// Takes `steps` steps of velocity Verlet of size `dt`, on the device and without copying
    /// anything between the device and the host but the tables of what is imposed, which are
    /// copied for as many steps at a time as they hold; and waits for them.
    ///
    /// An element that fails is not noticed until the steps are done, and then it is an error,
    /// and the state is of no use.
    pub fn steps(&self, dt: f64, steps: usize, schedule: &Schedule<'_>) -> Result<(), GpuError> {
        let dynamics = self.started()?;
        self.check(dynamics, schedule)?;
        let count = u32::try_from(3 * self.nodes)
            .map_err(|_| GpuError::Cuda("a model is too large for a device".to_string()))?;
        self.forget_failures()?;
        let mut done = 0;
        while done < steps {
            let rows = (steps - done).min(dynamics.rows);
            self.upload_tables(dynamics, schedule, dynamics.row.get() + 1, rows)?;
            for row in 0..rows {
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
                self.launch_targets(dynamics, row as u32)?;
                self.launch_forces()?;
                self.launch_loads(dynamics, row as u32)?;
                self.launch_accelerate(dt, true)?;
            }
            dynamics.row.set(dynamics.row.get() + rows);
            done += rows;
        }
        self.device.synchronize()?;
        self.failed()
    }
    fn started(&self) -> Result<&Dynamics, GpuError> {
        self.dynamics
            .as_ref()
            .ok_or_else(|| GpuError::Invalid("a run is started before it is stepped".to_string()))
    }
    /// That the schedule has what the run imposes.
    fn check(&self, dynamics: &Dynamics, schedule: &Schedule<'_>) -> Result<(), GpuError> {
        if dynamics.moved.is_some() && schedule.targets.is_none() {
            return Err(GpuError::Invalid(
                "the schedule has no targets for the coordinates that are moved".to_string(),
            ));
        }
        let weighted = dynamics
            .loads
            .iter()
            .filter(|load| matches!(load, LoadOnDevice::Weighted { .. }))
            .count();
        if schedule.components.len() != weighted {
            return Err(GpuError::Invalid(format!(
                "the schedule has the components of {} loads, and {weighted} are weighted",
                schedule.components.len()
            )));
        }
        Ok(())
    }
    /// Gives the device the rows of the tables from `first`, `rows` of them, which start at row 0
    /// of the tables.
    fn upload_tables(
        &self,
        dynamics: &Dynamics,
        schedule: &Schedule<'_>,
        first: usize,
        rows: usize,
    ) -> Result<(), GpuError> {
        if let (Some(moved), Some(targets)) = (&dynamics.moved, schedule.targets) {
            let mut values = Vec::with_capacity(moved.count as usize * rows);
            for row in first..first + rows {
                let targets = targets(row);
                if targets.len() != moved.count as usize {
                    return Err(GpuError::Invalid(format!(
                        "the schedule has {} targets, and {} coordinates are moved",
                        targets.len(),
                        moved.count
                    )));
                }
                values.extend(targets);
            }
            moved.table.upload_front(&values)?;
        }
        let mut components = schedule.components.iter();
        for load in &dynamics.loads {
            if let LoadOnDevice::Weighted { table, .. } = load {
                let components = components
                    .next()
                    .expect("the schedule has the components of every weighted load");
                let values = (first..first + rows)
                    .flat_map(components)
                    .collect::<Vec<_>>();
                table.upload_front(&values)?;
            }
        }
        Ok(())
    }
    /// Starts the setting of the coordinates that are moved to what they are held at in `row` of
    /// the tables.
    fn launch_targets(&self, dynamics: &Dynamics, row: u32) -> Result<(), GpuError> {
        match &dynamics.moved {
            Some(moved) => self.device.launch(
                &self.set_targets_kernel,
                moved.count as usize,
                THREADS_PER_BLOCK,
                &[
                    &moved.count,
                    &moved.dofs,
                    &moved.table,
                    &row,
                    &self.coordinates,
                ],
            ),
            None => Ok(()),
        }
    }
    /// Starts the subtracting of the loads, in order, from the forces, with `row` of the tables.
    fn launch_loads(&self, dynamics: &Dynamics, row: u32) -> Result<(), GpuError> {
        let count = (3 * self.nodes) as u32;
        let nodes = self.nodes as u32;
        for load in &dynamics.loads {
            match load {
                LoadOnDevice::Constant(forces) => self.device.launch(
                    &self.subtract_constant_kernel,
                    3 * self.nodes,
                    THREADS_PER_BLOCK,
                    &[&count, forces, &self.nodal_forces],
                )?,
                LoadOnDevice::Weighted { weights, table } => self.device.launch(
                    &self.subtract_weighted_kernel,
                    self.nodes,
                    THREADS_PER_BLOCK,
                    &[&nodes, weights, table, &row, &self.nodal_forces],
                )?,
            }
        }
        Ok(())
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
    ///
    /// The nodes are made for them, which costs more than putting them into nodes that are kept,
    /// as [`dynamics_state_into`](Self::dynamics_state_into) does.
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
        let mut state = (
            NodalCoordinates::<3>::zero(self.nodes),
            NodalVelocities::<3>::zero(self.nodes),
            NodalAccelerations::<3>::zero(self.nodes),
        );
        self.dynamics_state_into(&mut state.0, &mut state.1, &mut state.2)?;
        Ok(state)
    }
    /// The coordinates, velocities and accelerations of the run of dynamics, put into the nodes
    /// that are given, which have the nodes of the model, and which what was in them does not
    /// matter to.
    ///
    /// Memory that was not used before is slow to touch once a device is open, so a run that
    /// takes its state many times is better to keep the nodes that it takes it into.
    pub fn dynamics_state_into(
        &self,
        coordinates: &mut NodalCoordinates<3>,
        velocities: &mut NodalVelocities<3>,
        accelerations: &mut NodalAccelerations<3>,
    ) -> Result<(), GpuError> {
        let dynamics = self.started()?;
        for nodes in [coordinates.len(), velocities.len(), accelerations.len()] {
            assert_eq!(
                nodes, self.nodes,
                "a model is run with the nodes that it has"
            );
        }
        self.coordinates.download(coordinates.components_mut())?;
        dynamics.velocities.download(velocities.components_mut())?;
        dynamics
            .accelerations
            .download(accelerations.components_mut())
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
        let mut forces = NodalForcesSolid::<3>::zero(self.nodes);
        self.download_into(&mut forces)?;
        Ok(forces)
    }
    /// The nodal forces that the last computation made, put into the nodes that are given, which
    /// have the nodes of the model, and which what was in them does not matter to.
    pub fn download_into(&self, forces: &mut NodalForcesSolid<3>) -> Result<(), GpuError> {
        assert_eq!(
            forces.len(),
            self.nodes,
            "a model is run with the nodes that it has"
        );
        self.nodal_forces.download(forces.components_mut())
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
        let mut forces = NodalForcesSolid::<3>::zero(self.nodes);
        self.nodal_forces_into(model, coordinates, &mut forces)?;
        Ok(forces)
    }
    /// The nodal forces of the model for the coordinates, put into the nodes that are given, as
    /// [`nodal_forces`](Self::nodal_forces) does, without making nodes for them. A run that
    /// takes the forces many times is better to keep the nodes that it takes them into.
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
        Ok(self.download_into(forces)?)
    }
}
