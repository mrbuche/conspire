#[cfg(test)]
mod test;

use super::StabilizedElement;
use crate::math::{Quantity, Tensor};
use core::{array::from_fn, iter::once};

impl<const N: usize> StabilizedElement<N> {
    /// The number of values that describe an element for a device: the components of the mean
    /// gradient of each node, then the volume, then the hourglass matrix by rows.
    ///
    /// The volumes of the nodes are not among them, as the forces do not use them.
    pub const VALUES: usize = 3 * N + 1 + N * N;
    /// The values of the element, in the order that [`from_values`](Self::from_values) takes them.
    pub fn values(&self) -> Vec<f64> {
        let mut values = Vec::with_capacity(Self::VALUES);
        values.extend(
            self.gradient_vectors[0]
                .iter()
                .flat_map(|gradient_vector| gradient_vector.iter())
                .map(|component| component.value()),
        );
        values.push(self.integration_weights[0].value());
        values.extend(
            self.hourglass_matrix
                .iter()
                .flat_map(|row| row.iter())
                .map(|entry| entry.value()),
        );
        values
    }
    /// The element that has the values, which `value` gives by their position, as
    /// [`values`](Self::values) makes them, with volumes of the nodes that are nothing.
    pub fn from_values(value: impl Fn(usize) -> f64) -> Self {
        Self {
            gradient_vectors: once(
                (0..N)
                    .map(|a| from_fn(|k| value(3 * a + k)).into())
                    .collect(),
            )
            .collect(),
            integration_weights: [Quantity::new(value(3 * N))].into(),
            hourglass_matrix: from_fn(|a| from_fn(|b| Quantity::new(value(3 * N + 1 + a * N + b)))),
            nodal_volumes: (0..N).map(|_| Quantity::new(0.0)).collect(),
        }
    }
}
