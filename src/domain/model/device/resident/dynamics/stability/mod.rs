use crate::{
    domain::{
        ModelBound,
        stability::{Checked, CourantFriedrichsLewy, Unchecked},
    },
    math::Scalar,
};

/// Whether a run of dynamics on a device checks that its time step is stable, which it does
/// only with the [Courant-Friedrichs-Lewy](CourantFriedrichsLewy) estimate of the moduli that
/// [bound the model](ModelBound).
pub trait DeviceStability {
    /// How many steps there are from one check to the next, and the fraction of the stability
    /// limit that a step may be, if the run is checked.
    fn device_check(&self) -> Option<(usize, Scalar)>;
}

impl DeviceStability for Unchecked {
    fn device_check(&self) -> Option<(usize, Scalar)> {
        None
    }
}

impl DeviceStability for Checked<CourantFriedrichsLewy<ModelBound>> {
    fn device_check(&self) -> Option<(usize, Scalar)> {
        Some((self.interval, self.safety))
    }
}
