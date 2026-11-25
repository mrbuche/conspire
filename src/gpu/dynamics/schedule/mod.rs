#[cfg(test)]
mod test;

use crate::{gpu::GpuError, math::Quantity, units::Time};

/// What is imposed on a run of dynamics as a function of the step, which is the row of a table:
/// row 0 is the start, and row `r` is the state after `r` steps.
///
/// A device is given as many rows as it takes steps at a time, not those of the whole run. What a
/// schedule does not give is none, so that a run that needs it says so, and a run that does not
/// needs no schedule, for which `()` is one.
pub trait Schedule {
    /// The time of a row, which the external forces that are a function of time are asked for.
    fn time(&self, _row: usize) -> Option<Quantity<Time>> {
        None
    }
    /// What each of the coordinates that are moved is set to at a row, in the order of
    /// [`Imposed::moved`](super::Imposed::moved).
    fn targets(&self, _row: usize) -> Option<Vec<f64>> {
        None
    }
    /// That this gives what a run needs of it: targets if any coordinates are moved, and times if
    /// any of the external forces are a function of time, of which there are `weighted`.
    fn check(&self, moved: bool, weighted: usize) -> Result<(), GpuError> {
        if moved && self.targets(0).is_none() {
            return Err(GpuError::Invalid(
                "the schedule has no targets for the coordinates that are moved".to_string(),
            ));
        }
        if weighted > 0 && self.time(0).is_none() {
            return Err(GpuError::Invalid(format!(
                "the schedule has no times for the {weighted} loads that are a function of time"
            )));
        }
        Ok(())
    }
    /// The times of the rows `first..first + rows`.
    fn times(&self, first: usize, rows: usize) -> Result<Vec<Quantity<Time>>, GpuError> {
        (first..first + rows)
            .map(|row| {
                self.time(row).ok_or_else(|| {
                    GpuError::Invalid(format!("the schedule has no time for the row {row}"))
                })
            })
            .collect()
    }
    /// What the moved coordinates are set to in the rows `first..first + rows`, row by row, as
    /// `count` of them are moved. No coordinates moved is no values.
    fn target_rows(&self, count: usize, first: usize, rows: usize) -> Result<Vec<f64>, GpuError> {
        if count == 0 {
            return Ok(Vec::new());
        }
        let mut values = Vec::with_capacity(count * rows);
        for row in first..first + rows {
            let targets = self.targets(row).ok_or_else(|| {
                GpuError::Invalid(format!("the schedule has no targets for the row {row}"))
            })?;
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
}

impl Schedule for () {}
