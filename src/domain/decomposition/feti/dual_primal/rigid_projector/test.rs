use super::RigidProjector;
use crate::math::assert::{Assert, AssertionError};
use crate::math::{Tensor, Vector};

fn vector(entries: &[f64]) -> Vector {
    entries.iter().copied().collect()
}

fn projector() -> RigidProjector {
    RigidProjector::from_columns(vec![
        vector(&[1.0, 0.0, 1.0, 0.0]),
        vector(&[0.0, 1.0, 1.0, 1.0]),
    ])
    .unwrap()
}

fn transposed(projector: &RigidProjector, v: &Vector) -> Vector {
    projector.restrict(v)
}

#[test]
fn a_projected_vector_has_nothing_along_the_columns() {
    let projector = projector();
    let projected = projector.project(&vector(&[3.0, -1.0, 2.0, 5.0]));
    transposed(&projector, &projected)
        .iter()
        .for_each(|&entry| assert!(entry.abs() < 1e-12));
}

#[test]
fn projecting_twice_is_projecting_once() -> Result<(), AssertionError> {
    let projector = projector();
    let once = projector.project(&vector(&[3.0, -1.0, 2.0, 5.0]));
    let twice = projector.project(&once);
    (0..4).try_for_each(|row| Assert::within(1e-12).eq_within_tols(once[row], &twice[row]))?;
    Ok(())
}

#[test]
fn the_particular_multipliers_satisfy_the_constraint() -> Result<(), AssertionError> {
    let projector = projector();
    let e = vector(&[2.0, -3.0]);
    let lambda = projector.particular(&e, 4);
    let restricted = transposed(&projector, &lambda);
    (0..2).try_for_each(|row| Assert::within(1e-12).eq_within_tols(restricted[row], &e[row]))?;
    Ok(())
}

#[test]
fn amplitudes_recover_a_combination_of_the_columns() -> Result<(), AssertionError> {
    let projector = projector();
    let combination = projector.particular(&vector(&[1.0, 0.0]), 4);
    let amplitudes = projector.amplitudes(&combination);
    let expected = projector.expand(&amplitudes, 4);
    (0..4).try_for_each(|row| {
        Assert::within(1e-12).eq_within_tols(combination[row], &expected[row])
    })?;
    Ok(())
}

#[test]
fn dependent_columns_are_refused() {
    assert!(
        RigidProjector::from_columns(vec![vector(&[1.0, 2.0]), vector(&[-1.0, -2.0])]).is_none()
    );
}

#[test]
fn no_columns_project_nothing() {
    let projector = RigidProjector::from_columns(Vec::new()).unwrap();
    assert_eq!(projector.len(), 0);
    let v = vector(&[1.0, 2.0]);
    assert_eq!(projector.project(&v)[1], 2.0);
}
