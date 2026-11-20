#[cfg(test)]
mod test;

/// A constitutive model as plain numbers: its parameters, in `f64`.
pub trait FromParameters
where
    Self: Sized,
{
    /// The name of the constitutive model in snake case.
    const NAME: &'static str;
    /// The parameters of the model, in the order that [`from_parameters`](Self::from_parameters) takes them.
    fn parameters(&self) -> Vec<f64>;
    /// The model from the parameters, which must be as many as [`parameters`](Self::parameters) gives.
    fn from_parameters(parameters: &[f64]) -> Self;
}
