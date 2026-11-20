#[cfg(test)]
mod test;

use super::StabilizedElement;
use crate::{domain::block::device::DeviceKind, fem::element::DeviceElement};

impl DeviceElement<1, 8> for StabilizedElement<8> {
    const KIND: DeviceKind = DeviceKind::Stabilized;
    const VALUES: usize = StabilizedElement::<8>::VALUES;
    fn shared() -> (Vec<f64>, Vec<f64>) {
        (Vec::new(), Vec::new())
    }
    fn values(&self) -> Vec<f64> {
        StabilizedElement::values(self)
    }
}
