#[cfg(test)]
mod test;

use super::GpuError;

/// The flags that a run of dynamics on a device takes for its held degrees of freedom: one for
/// each of the `dofs` degrees of freedom of a model, as `3 * node + component`, which is one if
/// it is held and zero if it is not.
///
/// A degree of freedom that the model does not have is an error.
pub fn held_flags(dofs: usize, held: &[usize]) -> Result<Vec<u32>, GpuError> {
    let mut flags = vec![0; dofs];
    for &dof in held {
        *flags.get_mut(dof).ok_or_else(|| {
            GpuError::Invalid(format!(
                "the degree of freedom {dof} is held, and the model has {dofs} of them"
            ))
        })? = 1;
    }
    Ok(flags)
}
