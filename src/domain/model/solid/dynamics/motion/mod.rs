#[cfg(test)]
mod test;

use crate::{
    domain::{
        NodalCoordinates, NodalCoordinatesHistory, NodalVelocities, Stepping,
        boundary::{ConstrainAt, Solid},
        solid::mass::MassMatrix,
    },
    math::{
        Quantity, Tensor,
        integrate::{IntegrationError, Times},
        optimize::{EqualityConstraint, EqualityConstraints, Prescription},
    },
    units::Time,
};
use alloc::collections::BTreeSet;

/// The degrees of freedom of a solid held in place, and the motion of those that are moved.
///
/// The conditions are on the coordinates of the nodes, indexed as `3 * node + component`. What
/// is held at a distance from where it starts is held at that distance from the coordinates
/// the motion starts at, whatever the step it is held in. Fixed degrees of freedom have no
/// velocity, and those that are moved keep the one they are given, which a scheme that
/// moves them itself uses, and an explicit one does not report.
pub(super) struct Motion<'c, C> {
    conditions: &'c C,
    anchor: NodalCoordinates<3>,
    moving: Vec<usize>,
}

impl<'c, C> Motion<'c, C>
where
    C: ConstrainAt<Solid<3>>,
{
    /// The masses with the held degrees of freedom removed, the velocities with them removed,
    /// and the motion, once the coordinates are moved to where the motion starts.
    pub(super) fn held<M>(
        masses: &M,
        conditions: &'c C,
        initial_time: Quantity<Time>,
        coordinates: &mut NodalCoordinates<3>,
        velocities: NodalVelocities<3>,
        stepping: Stepping,
    ) -> Result<(M::Inverse, NodalVelocities<3>, Self), IntegrationError>
    where
        M: MassMatrix<3>,
    {
        let (held, velocities, motion) =
            Self::start(conditions, initial_time, coordinates, velocities, stepping)?;
        let inverse = masses
            .inverse(&held)
            .map_err(|error| IntegrationError::Intermediate(error.to_string()))?;
        Ok((inverse, velocities, motion))
    }
    /// The degrees of freedom that are held, which are those fixed and those moved, in ascending
    /// order, the velocities with the fixed ones removed, and the motion, once the coordinates are
    /// moved to where the motion starts.
    ///
    /// This is [`held`](Self::held) without the masses, for a run that divides by them itself.
    pub(super) fn start(
        conditions: &'c C,
        initial_time: Quantity<Time>,
        coordinates: &mut NodalCoordinates<3>,
        mut velocities: NodalVelocities<3>,
        stepping: Stepping,
    ) -> Result<(Vec<usize>, NodalVelocities<3>, Self), IntegrationError> {
        let mut fixed = BTreeSet::new();
        let mut moving = BTreeSet::new();
        for constraint in conditions.constraint_at(initial_time) {
            match constraint {
                EqualityConstraint::None => {}
                EqualityConstraint::Fixed(indices) => fixed.extend(indices),
                EqualityConstraint::Prescribed(entries) => {
                    entries.into_iter().for_each(|(index, prescription)| {
                        if prescription == Prescription::Initial {
                            fixed.insert(index);
                        } else {
                            moving.insert(index);
                        }
                    })
                }
                EqualityConstraint::Multipliers(entries) if stepping == Stepping::Implicit => {
                    entries.into_iter().for_each(|(index, prescription)| {
                        if prescription == Prescription::Initial {
                            fixed.insert(index);
                        } else {
                            moving.insert(index);
                        }
                    })
                }
                EqualityConstraint::Multipliers(_) => {
                    return Err(IntegrationError::Intermediate(
                        "Constraints held by multipliers are not supported by explicit dynamics."
                            .to_string(),
                    ));
                }
                EqualityConstraint::Linear(..) => {
                    return Err(IntegrationError::Intermediate(
                        "Linear constraints are not supported by dynamics.".to_string(),
                    ));
                }
            }
        }
        moving.retain(|dof| !fixed.contains(dof));
        let held: Vec<usize> = fixed.union(&moving).copied().collect();
        if let Some(dof) = held.iter().find(|&&dof| dof >= 3 * coordinates.len()) {
            return Err(IntegrationError::Intermediate(format!(
                "The degree of freedom {dof} that a boundary condition holds is not in the model."
            )));
        }
        fixed
            .iter()
            .for_each(|&dof| velocities[dof / 3][dof % 3] = Default::default());
        let motion = Self {
            conditions,
            anchor: coordinates.clone(),
            moving: moving.into_iter().collect(),
        };
        motion.apply(initial_time, coordinates);
        Ok((held, velocities, motion))
    }
    /// The constraints at a time, with what is held at a distance from where it starts held at
    /// that distance from the coordinates the motion starts at.
    pub(super) fn constraint_at(&self, time: Quantity<Time>) -> EqualityConstraints {
        EqualityConstraint::from_reference(self.conditions.constraint_at(time), &self.anchor)
    }
    /// Where a degree of freedom is when the motion starts.
    fn anchored(&self, dof: usize) -> f64 {
        self.anchor[dof / 3][dof % 3].value()
    }
    /// Puts into `targets` what each of the moved degrees of freedom is held at, at a time, in
    /// their order, which is ascending: where it starts, unless a condition holds it elsewhere,
    /// the last one that does.
    pub(super) fn targets_into(&self, time: Quantity<Time>, targets: &mut [f64]) {
        assert_eq!(
            targets.len(),
            self.moving.len(),
            "there is a target for each degree of freedom that is moved"
        );
        targets
            .iter_mut()
            .zip(&self.moving)
            .for_each(|(target, &dof)| *target = self.anchored(dof));
        let mut next = 0;
        self.conditions
            .prescribed_at(time, &mut |dof, prescription| {
                let found = if self.moving.get(next) == Some(&dof) {
                    Ok(next)
                } else {
                    self.moving.binary_search(&dof)
                };
                if let Ok(index) = found {
                    match prescription {
                        Prescription::Initial => {}
                        Prescription::Offset(offset) => {
                            targets[index] = self.anchored(dof) + offset
                        }
                        Prescription::Absolute(value) => targets[index] = value,
                    }
                    next = index + 1;
                }
            });
    }
    /// Moves the degrees of freedom that are moved to where they are held at a time.
    pub(super) fn apply(&self, time: Quantity<Time>, coordinates: &mut NodalCoordinates<3>) {
        let mut targets = vec![0.0; self.moving.len()];
        self.targets_into(time, &mut targets);
        self.moving.iter().zip(targets).for_each(|(&dof, target)| {
            coordinates[dof / 3][dof % 3] = Quantity::new(target);
        });
    }
    /// Gives the degrees of freedom that are moved the motion they are held at in the history.
    pub(super) fn record(&self, times: &Times, coordinates: &mut NodalCoordinatesHistory<3>) {
        times
            .iter()
            .take(coordinates.len())
            .enumerate()
            .for_each(|(k, &time)| {
                self.apply(time, &mut coordinates[k]);
            });
    }
}
