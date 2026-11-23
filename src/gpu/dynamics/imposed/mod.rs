#[cfg(test)]
mod test;

use crate::gpu::GpuError;

/// The values of the tables of what is imposed that a device holds, which a few million is.
const TABLE_VALUES: usize = 1 << 22;

/// The most rows of a table that a device holds, which a long run is given in turn.
const MAX_ROWS: usize = 1024;

/// An external force on a run of dynamics, which is subtracted from the forces of the elements.
pub enum ImposedLoad {
    /// A force on each degree of freedom, as `3 * node + component`, which does not change.
    Constant(Vec<f64>),
    /// The weight of each node, whose force is the weight times three components that are a
    /// function of the step, which the [`Schedule`] gives.
    Weighted(Vec<f64>),
}

/// What is imposed on a run of dynamics, which does not change from step to step.
///
/// The loads are subtracted from the forces in the order that they are given, as the host
/// subtracts them, because the order is part of the last bits.
#[derive(Default)]
pub struct Imposed {
    /// The degrees of freedom, as `3 * node + component`, that are held where they are: they have
    /// no acceleration.
    pub fixed: Vec<usize>,
    /// The degrees of freedom whose coordinates are set at every step, which have no acceleration
    /// either.
    pub moved: Vec<usize>,
    /// The external forces.
    pub loads: Vec<ImposedLoad>,
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
    /// That this is imposed on a model of `nodes` nodes: that the degrees of freedom are its, and
    /// that each force has a value for each of its degrees of freedom or its nodes.
    pub fn check(&self, nodes: usize) -> Result<(), GpuError> {
        self.held_flags(nodes)?;
        for load in &self.loads {
            match load {
                ImposedLoad::Constant(forces) if forces.len() != 3 * nodes => {
                    return Err(GpuError::Invalid(format!(
                        "a load has {} values, and the model has {} degrees of freedom",
                        forces.len(),
                        3 * nodes
                    )));
                }
                ImposedLoad::Weighted(weights) if weights.len() != nodes => {
                    return Err(GpuError::Invalid(format!(
                        "a load has {} weights, and the model has {nodes} nodes",
                        weights.len()
                    )));
                }
                _ => {}
            }
        }
        Ok(())
    }
    /// The number of loads that are a function of the step.
    pub fn weighted(&self) -> usize {
        self.loads
            .iter()
            .filter(|load| matches!(load, ImposedLoad::Weighted(_)))
            .count()
    }
    /// How many rows of the tables of what is a function of the step a device holds at a time:
    /// as many as fit in a table of a few million values, and a thousand at most. It is given the
    /// next rows when they are used up, so a long run does not need a table as long.
    pub fn rows(&self) -> usize {
        (TABLE_VALUES / self.moved.len().max(3)).clamp(1, MAX_ROWS)
    }
}

/// What is imposed on a run of dynamics as a function of the step, which is the row of a table:
/// row 0 is the start, and row `r` is the state after `r` steps.
///
/// A device is given as many rows as it takes steps at a time, not those of the whole run.
#[derive(Default)]
pub struct Schedule<'a> {
    /// What each of the moved degrees of freedom is set to, in the order of [`Imposed::moved`].
    pub targets: Option<&'a dyn Fn(usize) -> Vec<f64>>,
    /// The components of each weighted load, in the order of the loads.
    pub components: Vec<&'a dyn Fn(usize) -> [f64; 3]>,
}

impl Schedule<'_> {
    /// That this is a schedule of what is imposed: targets if any coordinates are moved, and the
    /// components of each load that is a function of the step.
    pub fn check(&self, imposed: &Imposed) -> Result<(), GpuError> {
        self.check_for(!imposed.moved.is_empty(), imposed.weighted())
    }
    /// The same of a run that has coordinates that are moved if `moved`, and `weighted` loads
    /// that are a function of the step.
    pub fn check_for(&self, moved: bool, weighted: usize) -> Result<(), GpuError> {
        if moved && self.targets.is_none() {
            return Err(GpuError::Invalid(
                "the schedule has no targets for the coordinates that are moved".to_string(),
            ));
        }
        if self.components.len() != weighted {
            return Err(GpuError::Invalid(format!(
                "the schedule has the components of {} loads, and {weighted} are weighted",
                self.components.len()
            )));
        }
        Ok(())
    }
    /// What the moved coordinates are set to in the rows `first..first + rows`, row by row, as
    /// `count` of them are moved. No coordinates moved is no values.
    pub fn target_rows(
        &self,
        count: usize,
        first: usize,
        rows: usize,
    ) -> Result<Vec<f64>, GpuError> {
        let Some(targets) = self.targets.filter(|_| count > 0) else {
            return Ok(Vec::new());
        };
        let mut values = Vec::with_capacity(count * rows);
        for row in first..first + rows {
            let targets = targets(row);
            if targets.len() != count {
                return Err(GpuError::Invalid(format!(
                    "the schedule has {} targets, and {count} coordinates are moved",
                    targets.len()
                )));
            }
            values.extend(targets);
        }
        Ok(values)
    }
    /// The components of the weighted load `load`, which is counted among the weighted loads, in
    /// the rows `first..first + rows`, three to a row.
    pub fn component_rows(&self, load: usize, first: usize, rows: usize) -> Vec<f64> {
        (first..first + rows)
            .flat_map(self.components[load])
            .collect()
    }
}
