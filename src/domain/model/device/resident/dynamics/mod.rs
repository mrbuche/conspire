use super::{Resident, THREADS_PER_BLOCK, flatten};
use crate::{
    domain::{
        NodalAccelerations, NodalCoordinates, NodalVelocities, solid::mass::NodalLumpedMasses,
    },
    gpu::{Application, Buffer, Function, GpuError, held_flags, kernels::ptx},
    math::Tensor,
};

/// The state of a run of dynamics on the device, and the kernels that take its steps.
pub(super) struct Dynamics {
    kick_drift: Function,
    accelerate: Function,
    velocities: Buffer<f64>,
    accelerations: Buffer<f64>,
    masses: Buffer<f64>,
    held: Buffer<u32>,
}

impl Resident {
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
        let dofs = 3 * self.nodes;
        let flags = held_flags(dofs, held)?;
        let mut velocities = flatten(velocities);
        held.iter().for_each(|&dof| velocities[dof] = 0.0);
        let (kick_drift, accelerate) = match self.dynamics.take() {
            Some(dynamics) => (dynamics.kick_drift, dynamics.accelerate),
            None => {
                let module = self
                    .device
                    .module_from_ptx(&ptx(Application::Dynamics, self.fused)?)?;
                (
                    module.function("kick_drift")?,
                    module.function("accelerate")?,
                )
            }
        };
        self.set_coordinates(coordinates)?;
        self.dynamics = Some(Dynamics {
            kick_drift,
            accelerate,
            velocities: self.device.buffer_of(&velocities)?,
            accelerations: self.device.buffer(dofs)?,
            masses: self
                .device
                .buffer_of(&masses.iter().map(|mass| mass.value()).collect::<Vec<_>>())?,
            held: self.device.buffer_of(&flags)?,
        });
        self.forget_errors()?;
        self.launch_forces()?;
        self.launch_accelerate(0.0, false)?;
        self.device.synchronize()?;
        self.errors()
    }
    /// Takes `steps` steps of velocity Verlet of size `dt`, on the device and without copying
    /// anything between the device and the host, and waits for them.
    ///
    /// An element that has an error is not noticed until the steps are done, and then it is an
    /// error, and the state is of no use.
    pub fn steps(&self, dt: f64, steps: usize) -> Result<(), GpuError> {
        let dynamics = self.started()?;
        let count = self.dofs()?;
        self.forget_errors()?;
        for _ in 0..steps {
            self.device.launch(
                &dynamics.kick_drift,
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
            Ok(values.as_chunks().0.to_vec())
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
    /// Starts the accelerations that balance the forces,
    /// and the velocities another half step on if `kick`.
    fn launch_accelerate(&self, dt: f64, kick: bool) -> Result<(), GpuError> {
        let dynamics = self.started()?;
        self.device.launch(
            &dynamics.accelerate,
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
