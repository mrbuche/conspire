#[cfg(test)]
mod test;

use super::Elastic;

/// An elastic constitutive model as a device runs it: by its parameters, as plain numbers.
///
/// A device builds the model again from the parameters, in `f64`, that the host gives it, and the
/// kernels it runs for a model are named for it, so a model that has this can be run on a device
/// without any more than a line to make its kernels. A model whose stress needs what a device does
/// not have, such as the exponential, does not have it.
pub trait DeviceModel
where
    Self: Elastic + Sized,
{
    /// The name of the model in the names of its kernels, which is in lowercase with underscores.
    const NAME: &'static str;
    /// The parameters of the model, in the order that [`from_parameters`](Self::from_parameters)
    /// takes them.
    fn parameters(&self) -> Vec<f64>;
    /// The model with the parameters, which must be as many as [`parameters`](Self::parameters)
    /// gives.
    fn from_parameters(parameters: &[f64]) -> Self;
}
