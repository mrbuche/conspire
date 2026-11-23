use super::{Resident, THREADS_PER_BLOCK};
use crate::{
    domain::{
        NodalAccelerations, NodalCoordinates, NodalVelocities, solid::mass::NodalLumpedMasses,
    },
    gpu::{Application, Buffer, Function, GpuError, Imposed, ImposedLoad, Schedule, kernels::ptx},
    math::Tensor,
};

/// The kernels of the steps.
struct Kernels {
    kick_drift: Function,
    accelerate: Function,
    set_targets: Function,
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

/// The state of a run of dynamics on the device, what is imposed on it,
/// and the kernels that take its steps.
pub(super) struct Dynamics {
    kernels: Kernels,
    velocities: Buffer<f64>,
    accelerations: Buffer<f64>,
    masses: Buffer<f64>,
    held: Buffer<u32>,
    moved: Option<Moved>,
    loads: Vec<LoadOnDevice>,
    /// What the kernel that makes the accelerations finds the loads by: whether each is a
    /// function of the step, and the addresses of its values and of its table.
    load_kinds: Buffer<u32>,
    load_data: Buffer<u64>,
    load_tables: Buffer<u64>,
    /// How many of the loads are a function of the step.
    weighted: usize,
    /// The rows that the tables hold.
    rows: usize,
    /// The row of the state of the run.
    row: usize,
}

impl Resident {
    /// Starts a run of dynamics from the coordinates and velocities, with the masses of the
    /// nodes, and what is imposed on it; `schedule` says what is imposed at the start.
    ///
    /// The velocities are those given: dynamics on the host has the fixed degrees of freedom
    /// without velocity, and the caller makes them so. The accelerations at the start are those
    /// of the forces on the coordinates, less the loads.
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
        imposed.check(self.nodes)?;
        schedule.check(imposed)?;
        let flags = imposed.held_flags(self.nodes)?;
        let rows = imposed.rows();
        let kernels = match self.dynamics.take() {
            Some(dynamics) => dynamics.kernels,
            None => {
                let module = self
                    .device
                    .module_from_ptx(&ptx(Application::Dynamics, self.fused)?)?;
                Kernels {
                    kick_drift: module.function("kick_drift")?,
                    accelerate: module.function("accelerate")?,
                    set_targets: module.function("set_targets")?,
                }
            }
        };
        let moved = if imposed.moved.is_empty() {
            None
        } else {
            let count = imposed.moved.len();
            let list = imposed
                .moved
                .iter()
                .map(|&dof| dof as u32)
                .collect::<Vec<_>>();
            Some(Moved {
                count: u32::try_from(count)
                    .map_err(|_| GpuError::Cuda("too many coordinates are moved".to_string()))?,
                dofs: self.device.buffer_of(&list)?,
                table: self.device.buffer(count * rows)?,
            })
        };
        let loads = imposed
            .loads
            .iter()
            .map(|load| -> Result<LoadOnDevice, GpuError> {
                Ok(match load {
                    ImposedLoad::Constant(forces) => {
                        LoadOnDevice::Constant(self.device.buffer_of(forces)?)
                    }
                    ImposedLoad::Weighted(weights) => LoadOnDevice::Weighted {
                        weights: self.device.buffer_of(weights)?,
                        table: self.device.buffer(3 * rows)?,
                    },
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let kinds = loads
            .iter()
            .map(|load| u32::from(matches!(load, LoadOnDevice::Weighted { .. })))
            .collect::<Vec<_>>();
        let data = loads
            .iter()
            .map(|load| match load {
                LoadOnDevice::Constant(forces) => forces.address(),
                LoadOnDevice::Weighted { weights, .. } => weights.address(),
            })
            .collect::<Vec<_>>();
        let tables = loads
            .iter()
            .map(|load| match load {
                LoadOnDevice::Constant(_) => 0,
                LoadOnDevice::Weighted { table, .. } => table.address(),
            })
            .collect::<Vec<_>>();
        self.set_coordinates(coordinates)?;
        self.dynamics = Some(Dynamics {
            kernels,
            velocities: self.device.buffer_of(velocities.as_flat())?,
            accelerations: self.device.buffer(3 * self.nodes)?,
            masses: self
                .device
                .buffer_of(&masses.iter().map(|mass| mass.value()).collect::<Vec<_>>())?,
            held: self.device.buffer_of(&flags)?,
            moved,
            loads,
            load_kinds: self.device.buffer_of(&kinds)?,
            load_data: self.device.buffer_of(&data)?,
            load_tables: self.device.buffer_of(&tables)?,
            weighted: imposed.weighted(),
            rows,
            row: 0,
        });
        self.upload_tables(schedule, 0, 1)?;
        self.forget_errors()?;
        self.launch_targets(0)?;
        self.launch_forces()?;
        self.launch_accelerate(0.0, false, 0)?;
        self.device.synchronize()?;
        self.errors()
    }
    /// Takes `steps` steps of velocity Verlet of size `dt`, on the device and without copying
    /// anything between the device and the host but the tables of what is imposed, which are
    /// copied for as many steps at a time as they hold; and waits for them.
    ///
    /// An element that has an error is not noticed until the steps are done, and then it is an
    /// error, and the state is of no use.
    pub fn steps(
        &mut self,
        dt: f64,
        steps: usize,
        schedule: &Schedule<'_>,
    ) -> Result<(), GpuError> {
        let dynamics = self.started()?;
        let (rows, count) = (dynamics.rows, self.dofs()?);
        schedule.check_for(dynamics.moved.is_some(), dynamics.weighted)?;
        self.forget_errors()?;
        let mut done = 0;
        while done < steps {
            let batch = (steps - done).min(rows);
            self.upload_tables(schedule, self.started()?.row + 1, batch)?;
            for row in 0..batch {
                let dynamics = self.started()?;
                self.device.launch(
                    &dynamics.kernels.kick_drift,
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
                self.launch_targets(row as u32)?;
                self.launch_forces()?;
                self.launch_accelerate(dt, true, row as u32)?;
            }
            if let Some(dynamics) = self.dynamics.as_mut() {
                dynamics.row += batch;
            }
            done += batch;
        }
        self.device.synchronize()?;
        self.errors()
    }
    /// The coordinates, velocities and accelerations of the run of dynamics, in new nodes.
    ///
    /// A run that takes them many times is better to keep nodes and use
    /// [`dynamics_state_into`](Self::dynamics_state_into): with a device open, memory that has
    /// not been used before is several times slower to touch than memory that has.
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
            NodalCoordinates::zero(self.nodes),
            NodalVelocities::zero(self.nodes),
            NodalAccelerations::zero(self.nodes),
        );
        self.dynamics_state_into(&mut state.0, &mut state.1, &mut state.2)?;
        Ok(state)
    }
    /// The coordinates, velocities and accelerations of the run of dynamics, put into nodes that
    /// have the nodes of the model, and whatever they held does not matter.
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
        self.coordinates.download(coordinates.as_flat_mut())?;
        dynamics.velocities.download(velocities.as_flat_mut())?;
        dynamics.accelerations.download(accelerations.as_flat_mut())
    }
    fn started(&self) -> Result<&Dynamics, GpuError> {
        self.dynamics
            .as_ref()
            .ok_or_else(|| GpuError::Invalid("a run is started before it is stepped".to_string()))
    }
    /// The number of degrees of freedom, as the kernels take it.
    fn dofs(&self) -> Result<u32, GpuError> {
        u32::try_from(3 * self.nodes)
            .map_err(|_| GpuError::Cuda("a model is too large for a device".to_string()))
    }
    /// Gives the device the rows of the tables from `first`, `rows` of them, which start at row 0
    /// of the tables.
    fn upload_tables(
        &self,
        schedule: &Schedule<'_>,
        first: usize,
        rows: usize,
    ) -> Result<(), GpuError> {
        let dynamics = self.started()?;
        if let Some(moved) = &dynamics.moved {
            moved.table.upload_front(&schedule.target_rows(
                moved.count as usize,
                first,
                rows,
            )?)?;
        }
        let mut weighted = 0;
        for load in &dynamics.loads {
            if let LoadOnDevice::Weighted { table, .. } = load {
                table.upload_front(&schedule.component_rows(weighted, first, rows))?;
                weighted += 1;
            }
        }
        Ok(())
    }
    /// Starts the setting of the coordinates that are moved to what they are held at in `row` of
    /// the tables.
    fn launch_targets(&self, row: u32) -> Result<(), GpuError> {
        let dynamics = self.started()?;
        match &dynamics.moved {
            Some(moved) => self.device.launch(
                &dynamics.kernels.set_targets,
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
    /// Starts the accelerations that balance the forces less the loads, which are subtracted in
    /// order with `row` of the tables, and the velocities another half step on if `kick`.
    fn launch_accelerate(&self, dt: f64, kick: bool, row: u32) -> Result<(), GpuError> {
        let dynamics = self.started()?;
        self.device.launch(
            &dynamics.kernels.accelerate,
            3 * self.nodes,
            THREADS_PER_BLOCK,
            &[
                &self.dofs()?,
                &self.nodal_forces,
                &dynamics.masses,
                &dynamics.held,
                &dynamics.accelerations,
                &dynamics.velocities,
                &dt,
                &u32::from(kick),
                &(dynamics.loads.len() as u32),
                &dynamics.load_kinds,
                &dynamics.load_data,
                &dynamics.load_tables,
                &row,
            ],
        )
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
