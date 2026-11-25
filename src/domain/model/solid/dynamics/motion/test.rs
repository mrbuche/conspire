use super::Motion;
use crate::{
    domain::{
        NodalCoordinates, NodalVelocities, Stepping,
        boundary::{BoundaryConditions, ConstrainAt, Displacement, Enforcement, Rows, Solid},
    },
    math::{
        Quantity,
        integrate::IntegrationError,
        optimize::{EqualityConstraint, EqualityConstraints, Prescription},
    },
    units::{Length, Time},
};

fn coordinates() -> NodalCoordinates<3> {
    (0..4)
        .map(|node| [0.0, 1.0, 2.0].map(|component| 10.0 * node as f64 + component))
        .collect::<Vec<_>>()
        .into()
}

fn velocities() -> NodalVelocities<3> {
    vec![[1.0; 3]; 4].into()
}

fn values<const N: usize>(coordinates: &NodalCoordinates<3>, dofs: [usize; N]) -> [f64; N] {
    dofs.map(|dof| coordinates[dof / 3][dof % 3].value())
}

fn refused(conditions: &impl ConstrainAt<Solid<3>>, stepping: Stepping) -> String {
    match Motion::start(
        conditions,
        Time::seconds(0.0),
        &mut coordinates(),
        velocities(),
        stepping,
    ) {
        Err(IntegrationError::Intermediate(message)) => message,
        Err(error) => panic!("{error:?}"),
        Ok(_) => panic!("the motion is started"),
    }
}

#[test]
fn what_is_held_is_given_in_order_and_what_is_fixed_has_no_velocity() {
    let conditions = BoundaryConditions::<Solid<3>>::new()
        .fix_nodes(Displacement::<0>, [2, 0])
        .prescribe_nodes(Displacement::<1>, [1], Length::meters(0.5));
    let mut found = coordinates();
    let (held, velocities, motion) = Motion::start(
        &conditions,
        Time::seconds(0.0),
        &mut found,
        velocities(),
        Stepping::Explicit,
    )
    .unwrap();
    assert_eq!(held, [0, 4, 6]);
    assert_eq!(motion.moving, [4]);
    assert_eq!(velocities[0][0].value(), 0.0);
    assert_eq!(velocities[2][0].value(), 0.0);
    assert_eq!(velocities[1][1].value(), 1.0);
    assert_eq!(velocities[0][1].value(), 1.0);
    assert_eq!(values(&found, [0, 4, 6]), [0.0, 11.5, 20.0]);
}

#[test]
fn the_targets_are_where_the_last_condition_holds_each_moved_degree_of_freedom() {
    let conditions = BoundaryConditions::<Solid<3>>::new()
        .fix_nodes(Displacement::<0>, [3])
        .prescribe_nodes_with(Displacement::<1>, [2, 0], |time| {
            Length::meters(time.value())
        })
        .prescribe_nodes(Displacement::<1>, [0], Length::meters(100.0))
        .prescribe_nodes(Displacement::<0>, [3, 1], Length::meters(5.0));
    let rows =
        Rows::<Solid<3>>::new(Enforcement::Elimination).hold(11, Prescription::Absolute(-1.0));
    let conditions = (conditions, rows);
    let mut found = coordinates();
    let (held, _, motion) = Motion::start(
        &conditions,
        Time::seconds(0.0),
        &mut found,
        velocities(),
        Stepping::Explicit,
    )
    .unwrap();
    assert_eq!(motion.moving, [1, 3, 7, 11]);
    assert_eq!(held, [1, 3, 7, 9, 11]);
    let mut targets = [f64::NAN; 4];
    motion.targets_into(Time::seconds(2.0), &mut targets);
    assert_eq!(targets, [101.0, 15.0, 23.0, -1.0]);
    motion.apply(Time::seconds(2.0), &mut found);
    assert_eq!(
        values(&found, [1, 3, 7, 11, 9]),
        [101.0, 15.0, 23.0, -1.0, 30.0]
    );
}

struct Lapsing;

impl ConstrainAt<Solid<3>> for Lapsing {
    fn constraint_at(&self, time: Quantity<Time>) -> EqualityConstraints {
        let prescription = if time.value() < 1.0 {
            Prescription::Offset(2.0)
        } else {
            Prescription::Initial
        };
        vec![EqualityConstraint::Prescribed(vec![(1, prescription)])]
    }
}

#[test]
fn a_moved_degree_of_freedom_that_is_not_held_elsewhere_is_where_it_starts() {
    let mut found = coordinates();
    let (_, _, motion) = Motion::start(
        &Lapsing,
        Time::seconds(0.0),
        &mut found,
        velocities(),
        Stepping::Explicit,
    )
    .unwrap();
    let mut targets = [f64::NAN];
    motion.targets_into(Time::seconds(0.5), &mut targets);
    assert_eq!(targets, [3.0]);
    motion.targets_into(Time::seconds(5.0), &mut targets);
    assert_eq!(targets, [1.0]);
}

#[test]
fn a_degree_of_freedom_that_is_not_in_the_model_is_refused() {
    for prescription in [Prescription::Initial, Prescription::Offset(1.0)] {
        let rows = Rows::<Solid<3>>::new(Enforcement::Elimination).hold(12, prescription);
        let message = refused(&rows, Stepping::Explicit);
        assert!(
            message.contains("12") && message.contains("not in the model"),
            "{message}"
        );
    }
}

#[test]
fn multipliers_are_refused_by_explicit_steps_and_taken_by_implicit_ones() {
    let rows = Rows::<Solid<3>>::new(Enforcement::Multipliers).hold(0, Prescription::Initial);
    let message = refused(&rows, Stepping::Explicit);
    assert!(message.contains("multipliers"), "{message}");
    assert!(
        Motion::start(
            &rows,
            Time::seconds(0.0),
            &mut coordinates(),
            velocities(),
            Stepping::Implicit,
        )
        .is_ok()
    );
}
