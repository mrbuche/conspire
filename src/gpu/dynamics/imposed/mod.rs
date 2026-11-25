#[cfg(test)]
mod test;

use crate::gpu::GpuError;

/// The values of the tables of what is imposed that a device holds, which a few million is.
const TABLE_VALUES: usize = 1 << 22;

/// The most rows of a table that a device holds, which a long run is given in turn.
const MAX_ROWS: usize = 1024;

/// What is imposed on a run of dynamics on the degrees of freedom, which does not change from step
/// to step. The external forces are given beside it, as `DeviceLoads`, and what is a function of
/// the step as a `Schedule`.
#[derive(Default)]
pub struct Imposed {
    /// The degrees of freedom, as `3 * node + component`, that are held where they are: they have
    /// no acceleration.
    pub fixed: Vec<usize>,
    /// The degrees of freedom whose coordinates are set at every step, which have no acceleration
    /// either.
    pub moved: Vec<usize>,
}

impl Imposed {
    /// The flags of the degrees of freedom of a model of `nodes` nodes that have no acceleration,
    /// which are those that are held and those that are moved: one if it has none, and zero if it
    /// has.
    ///
    /// A degree of freedom that the model does not have is an error.
    pub fn held_flags(&self, nodes: usize) -> Result<Vec<u32>, GpuError> {
        let dofs = 3 * nodes;
        let mut flags = vec![0; dofs];
        for &dof in self.fixed.iter().chain(&self.moved) {
            *flags.get_mut(dof).ok_or_else(|| {
                GpuError::Invalid(format!("the degree of freedom {dof} is not in the model"))
            })? = 1;
        }
        Ok(flags)
    }
    /// How many rows of the tables of what is a function of the step a device holds at a time:
    /// as many as fit in a table of a few million values, and a thousand at most. It is given the
    /// next rows when they are used up, so a long run does not need a table as long.
    pub fn rows(&self) -> usize {
        (TABLE_VALUES / self.moved.len().max(3)).clamp(1, MAX_ROWS)
    }
}
