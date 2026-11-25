use super::{Resident, THREADS_PER_BLOCK, flatten};
use crate::{
    domain::{
        NodalAccelerations, NodalCoordinates, NodalVelocities,
        boundary::{DeviceForce, DeviceLoads},
        solid::mass::NodalLumpedMasses,
    },
    gpu::{Application, Buffer, Function, GpuError, Imposed, Schedule, kernels::ptx},
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
    /// nodes, the degrees of freedom that are held or moved, the external forces, and what is a
    /// function of the step; `schedule` says what it is at the start.
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
        loads: &impl DeviceLoads,
        schedule: &impl Schedule,
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
        let flags = imposed.held_flags(self.nodes)?;
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
        schedule.check(!imposed.moved.is_empty(), weighted)?;
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
        self.set_coordinates(coordinates)?;
        self.dynamics = Some(Dynamics {
            kernels,
            velocities: self.device.buffer_of(&flatten(velocities))?,
            accelerations: self.device.buffer(3 * self.nodes)?,
            masses: self
                .device
                .buffer_of(&masses.iter().map(|mass| mass.value()).collect::<Vec<_>>())?,
            held: self.device.buffer_of(&flags)?,
            moved,
            loads: on_device,
            load_kinds: self.device.buffer_of(&kinds)?,
            load_data: self.device.buffer_of(&data)?,
            load_tables: self.device.buffer_of(&tables)?,
            weighted,
            rows,
            row: 0,
        });
        self.upload_tables(loads, schedule, 0, 1)?;
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
        loads: &impl DeviceLoads,
        schedule: &impl Schedule,
    ) -> Result<(), GpuError> {
        let dynamics = self.started()?;
        let (rows, count) = (dynamics.rows, self.dofs()?);
        if loads.device_weighted() != dynamics.weighted {
            return Err(GpuError::Invalid(format!(
                "the loads have {} that are a function of time, and the run started with {}",
                loads.device_weighted(),
                dynamics.weighted
            )));
        }
        schedule.check(dynamics.moved.is_some(), dynamics.weighted)?;
        self.forget_errors()?;
        let mut done = 0;
        while done < steps {
            let batch = (steps - done).min(rows);
            self.upload_tables(loads, schedule, self.started()?.row + 1, batch)?;
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
        loads: &impl DeviceLoads,
        schedule: &impl Schedule,
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
        if dynamics.weighted > 0 {
            let times = schedule.times(first, rows)?;
            let mut which = 0;
            for load in &dynamics.loads {
                if let LoadOnDevice::Weighted { table, .. } = load {
                    table.upload_front(&loads.device_component_rows(which, &times))?;
                    which += 1;
                }
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
