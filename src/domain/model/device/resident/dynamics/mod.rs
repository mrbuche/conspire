#[cfg(test)]
mod test;

use super::{Resident, THREADS_PER_BLOCK};
use crate::{
    domain::{
        NodalAccelerations, NodalCoordinates, NodalVelocities, Stepping,
        boundary::{ConstrainAt, DeviceForce, DeviceLoads, Solid},
        solid::{NodalForcesSolid, dynamics::motion::Motion, mass::NodalLumpedMasses},
    },
    gpu::{Application, Buffer, Device, Function, GpuError, kernels::ptx},
    math::{
        Quantity, Tensor,
        integrate::{FixedStep, VelocityVerlet},
    },
    units::Time,
};

const TABLE_VALUES: usize = 1 << 22;
const MAX_ROWS: usize = 1024;

fn table_rows(moved: usize) -> usize {
    (TABLE_VALUES / moved.max(3)).clamp(1, MAX_ROWS)
}

pub(super) struct Kernels {
    kick_drift: Function,
    accelerate: Function,
    set_targets: Function,
}

impl Kernels {
    fn new(device: &Device, fused: bool) -> Result<Self, GpuError> {
        let module = device.module_from_ptx(&ptx(Application::Dynamics, fused)?)?;
        Ok(Self {
            kick_drift: module.function("kick_drift")?,
            accelerate: module.function("accelerate")?,
            set_targets: module.function("set_targets")?,
        })
    }
}

/// The coordinates that are moved, and the rows of where they are held.
struct Moved {
    count: u32,
    dofs: Buffer<u32>,
    table: Buffer<f64>,
}

/// An external force that is subtracted from the forces of the elements.
enum LoadOnDevice {
    Constant(Buffer<f64>),
    Weighted {
        weights: Buffer<f64>,
        table: Buffer<f64>,
    },
}

/// A run of explicit dynamics on a device, by steps of velocity Verlet, whose state stays there.
///
/// It is made by [`Resident::explicit_dynamics`], and borrows the model on the device, the
/// boundary conditions and the external forces while it lives.
pub struct ExplicitRun<'a, C, L> {
    resident: &'a Resident,
    motion: Motion<'a, C>,
    loads: &'a L,
    start: Quantity<Time>,
    step: Quantity<Time>,
    velocities: Buffer<f64>,
    accelerations: Buffer<f64>,
    masses: Buffer<f64>,
    held: Buffer<u32>,
    moved: Option<Moved>,
    on_device: Vec<LoadOnDevice>,
    /// What the kernel that makes the accelerations finds the loads by: whether each is a
    /// function of the step, and the addresses of its values and of its table.
    load_kinds: Buffer<u32>,
    load_data: Buffer<u64>,
    load_tables: Buffer<u64>,
    /// The rows that the tables hold.
    rows: usize,
    /// The row of the state of the run, which is the number of steps that it has taken.
    row: usize,
}

impl Resident {
    /// Starts a run of explicit dynamics from the coordinates and velocities at a time, with the
    /// masses of the nodes, the external forces, and the boundary conditions, as the dynamics of
    /// the host takes them.
    ///
    /// The degrees of freedom that the conditions hold are those that they hold at the start: the
    /// fixed ones lose their velocity, and the coordinates of the moved ones are put where they
    /// are held. Constraints by multipliers and linear constraints are not taken. The
    /// accelerations at the start are those of the forces on the coordinates, less the loads.
    pub fn explicit_dynamics<'a, C, L>(
        &'a mut self,
        masses: &NodalLumpedMasses,
        integrator: &VelocityVerlet,
        time: Quantity<Time>,
        (coordinates, velocities): (&NodalCoordinates<3>, &NodalVelocities<3>),
        loads: &'a L,
        conditions: &'a C,
    ) -> Result<ExplicitRun<'a, C, L>, GpuError>
    where
        C: ConstrainAt<Solid<3>>,
        L: DeviceLoads,
    {
        for nodes in [masses.len(), coordinates.len(), velocities.len()] {
            assert_eq!(
                nodes, self.nodes,
                "a model is run with the nodes that it has"
            );
        }
        let step: Quantity<Time> = integrator.dt();
        if step.value() <= 0.0 || step.value().is_nan() {
            return Err(GpuError::Invalid(format!(
                "the step is {} seconds, which is not a time that passes",
                step.value()
            )));
        }
        let mut forces = Vec::new();
        loads.device_forces(&mut |force| forces.push(force));
        for force in &forces {
            force.check(self.nodes).map_err(GpuError::Invalid)?;
        }
        let weighted = loads.device_weighted();
        let given = forces
            .iter()
            .filter(|force| matches!(force, DeviceForce::Weighted(_)))
            .count();
        if given != weighted {
            return Err(GpuError::Invalid(format!(
                "the loads give {given} forces that are a function of time, and say that {weighted} are"
            )));
        }
        let mut moved_to_start = coordinates.clone();
        let (held, velocities, motion) = Motion::start(
            conditions,
            time,
            &mut moved_to_start,
            velocities.clone(),
            Stepping::Explicit,
        )
        .map_err(|error| GpuError::Invalid(error.to_string()))?;
        if self.kernels.is_none() {
            self.kernels = Some(Kernels::new(&self.device, self.fused)?);
        }
        let rows = table_rows(motion.moving().len());
        let moved = if motion.moving().is_empty() {
            None
        } else {
            let list = motion
                .moving()
                .iter()
                .map(|&dof| dof as u32)
                .collect::<Vec<_>>();
            Some(Moved {
                count: u32::try_from(list.len())
                    .map_err(|_| GpuError::Cuda("too many coordinates are moved".to_string()))?,
                dofs: self.device.buffer_of(&list)?,
                table: self.device.buffer(list.len() * rows)?,
            })
        };
        let mut flags = vec![0_u32; 3 * self.nodes];
        held.iter().for_each(|&dof| flags[dof] = 1);
        let on_device = forces
            .iter()
            .map(|force| -> Result<LoadOnDevice, GpuError> {
                Ok(match force {
                    DeviceForce::Constant(values) => {
                        LoadOnDevice::Constant(self.device.buffer_of(values)?)
                    }
                    DeviceForce::Weighted(weights) => LoadOnDevice::Weighted {
                        weights: self.device.buffer_of(weights)?,
                        table: self.device.buffer(3 * rows)?,
                    },
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let kinds = on_device
            .iter()
            .map(|load| u32::from(matches!(load, LoadOnDevice::Weighted { .. })))
            .collect::<Vec<_>>();
        let data = on_device
            .iter()
            .map(|load| match load {
                LoadOnDevice::Constant(forces) => forces.address(),
                LoadOnDevice::Weighted { weights, .. } => weights.address(),
            })
            .collect::<Vec<_>>();
        let tables = on_device
            .iter()
            .map(|load| match load {
                LoadOnDevice::Constant(_) => 0,
                LoadOnDevice::Weighted { table, .. } => table.address(),
            })
            .collect::<Vec<_>>();
        self.set_coordinates(&moved_to_start)?;
        let run = ExplicitRun {
            velocities: self.device.buffer_of(velocities.as_flat())?,
            accelerations: self.device.buffer(3 * self.nodes)?,
            masses: self
                .device
                .buffer_of(&masses.iter().map(|mass| mass.value()).collect::<Vec<_>>())?,
            held: self.device.buffer_of(&flags)?,
            moved,
            on_device,
            load_kinds: self.device.buffer_of(&kinds)?,
            load_data: self.device.buffer_of(&data)?,
            load_tables: self.device.buffer_of(&tables)?,
            rows,
            row: 0,
            resident: self,
            motion,
            loads,
            start: time,
            step,
        };
        run.upload_tables(&[time])?;
        run.resident.forget_errors()?;
        run.resident.launch_forces()?;
        run.launch_accelerate(0.0, false, 0)?;
        run.resident.device.synchronize()?;
        run.resident.errors()?;
        Ok(run)
    }
    /// The number of degrees of freedom, as the kernels take it.
    fn dofs(&self) -> Result<u32, GpuError> {
        u32::try_from(3 * self.nodes)
            .map_err(|_| GpuError::Cuda("a model is too large for a device".to_string()))
    }
    /// An error if an element has one in the computations since the errors were cleared.
    fn errors(&self) -> Result<(), GpuError> {
        match self.first_error()? {
            Some((block, record)) => Err(GpuError::Element {
                block,
                item: record.item(),
                point: record.point(),
            }),
            None => Ok(()),
        }
    }
}

impl<C, L> ExplicitRun<'_, C, L>
where
    C: ConstrainAt<Solid<3>>,
    L: DeviceLoads,
{
    /// Takes `steps` steps, on the device and without copying anything between the device and the
    /// host but the tables of what is a function of the step, which are copied for as many steps
    /// at a time as they hold; and waits for them.
    ///
    /// The times of the steps are those of the grid of fixed steps of the host, and each step is
    /// of the difference of its times, as on the host. An element that has an error is not noticed
    /// until the steps are done, and then it is an error, and the state is of no use.
    pub fn steps(&mut self, steps: usize) -> Result<(), GpuError> {
        let count = self.resident.dofs()?;
        self.resident.forget_errors()?;
        let mut done = 0;
        while done < steps {
            let batch = (steps - done).min(self.rows);
            let times = (self.row + 1..=self.row + batch)
                .map(|row| self.time(row))
                .collect::<Vec<_>>();
            self.upload_tables(&times)?;
            let mut previous = self.time(self.row);
            for (row, &time) in times.iter().enumerate() {
                let dt = (time - previous).value();
                self.resident.device.launch(
                    &self.kernels()?.kick_drift,
                    3 * self.resident.nodes,
                    THREADS_PER_BLOCK,
                    &[
                        &count,
                        &self.resident.coordinates,
                        &self.velocities,
                        &self.accelerations,
                        &dt,
                    ],
                )?;
                self.launch_targets(row as u32)?;
                self.resident.launch_forces()?;
                self.launch_accelerate(dt, true, row as u32)?;
                previous = time;
            }
            self.row += batch;
            done += batch;
        }
        self.resident.device.synchronize()?;
        self.resident.errors()
    }
    /// The time of a row of the grid of times, which is the state after as many steps.
    fn time(&self, row: usize) -> Quantity<Time> {
        self.start + self.step * (row as f64)
    }
    /// Gives the device what is a function of the step at the times, which are the rows of the
    /// tables from the first.
    fn upload_tables(&self, times: &[Quantity<Time>]) -> Result<(), GpuError> {
        if let Some(moved) = &self.moved {
            let count = moved.count as usize;
            let mut targets = vec![0.0; count * times.len()];
            targets
                .chunks_exact_mut(count)
                .zip(times)
                .for_each(|(row, &time)| self.motion.targets_into(time, row));
            moved.table.upload_front(&targets)?;
        }
        let mut which = 0;
        for load in &self.on_device {
            if let LoadOnDevice::Weighted { table, .. } = load {
                table.upload_front(&self.loads.device_component_rows(which, times))?;
                which += 1;
            }
        }
        Ok(())
    }
    /// Starts the setting of the coordinates that are moved to what they are held at in `row` of
    /// the tables.
    fn launch_targets(&self, row: u32) -> Result<(), GpuError> {
        match &self.moved {
            Some(moved) => self.resident.device.launch(
                &self.kernels()?.set_targets,
                moved.count as usize,
                THREADS_PER_BLOCK,
                &[
                    &moved.count,
                    &moved.dofs,
                    &moved.table,
                    &row,
                    &self.resident.coordinates,
                ],
            ),
            None => Ok(()),
        }
    }
    /// Starts the accelerations that balance the forces less the loads, which are subtracted in
    /// order with `row` of the tables, and the velocities another half step on if `kick`.
    fn launch_accelerate(&self, dt: f64, kick: bool, row: u32) -> Result<(), GpuError> {
        self.resident.device.launch(
            &self.kernels()?.accelerate,
            3 * self.resident.nodes,
            THREADS_PER_BLOCK,
            &[
                &self.resident.dofs()?,
                &self.resident.nodal_forces,
                &self.masses,
                &self.held,
                &self.accelerations,
                &self.velocities,
                &dt,
                &u32::from(kick),
                &(self.on_device.len() as u32),
                &self.load_kinds,
                &self.load_data,
                &self.load_tables,
                &row,
            ],
        )
    }
}

impl<C, L> ExplicitRun<'_, C, L> {
    /// The coordinates, velocities and accelerations of the run, in new nodes.
    ///
    /// A run that takes them many times is better to keep nodes and use
    /// [`state_into`](Self::state_into): taking the state into new nodes has been measured to
    /// take more than twice as long.
    pub fn state(
        &self,
    ) -> Result<
        (
            NodalCoordinates<3>,
            NodalVelocities<3>,
            NodalAccelerations<3>,
        ),
        GpuError,
    > {
        let nodes = self.resident.nodes;
        let mut state = (
            NodalCoordinates::zero(nodes),
            NodalVelocities::zero(nodes),
            NodalAccelerations::zero(nodes),
        );
        self.state_into(&mut state.0, &mut state.1, &mut state.2)?;
        Ok(state)
    }
    /// The coordinates, velocities and accelerations of the run, put into nodes that have the
    /// nodes of the model, and whatever they held does not matter.
    pub fn state_into(
        &self,
        coordinates: &mut NodalCoordinates<3>,
        velocities: &mut NodalVelocities<3>,
        accelerations: &mut NodalAccelerations<3>,
    ) -> Result<(), GpuError> {
        for nodes in [coordinates.len(), velocities.len(), accelerations.len()] {
            assert_eq!(
                nodes, self.resident.nodes,
                "a model is run with the nodes that it has"
            );
        }
        self.resident
            .coordinates
            .download(coordinates.as_flat_mut())?;
        self.velocities.download(velocities.as_flat_mut())?;
        self.accelerations.download(accelerations.as_flat_mut())
    }
    /// The nodal forces of the elements at the coordinates of the run, without the loads.
    pub fn forces(&self) -> Result<NodalForcesSolid<3>, GpuError> {
        self.resident.download()
    }
    fn kernels(&self) -> Result<&Kernels, GpuError> {
        self.resident
            .kernels
            .as_ref()
            .ok_or_else(|| GpuError::Invalid("a run has the kernels of its steps".to_string()))
    }
}
