use super::Failure;
use crate::{constitutive::ConstitutiveError, fem::element::FiniteElementError};

#[test]
fn a_failure_is_the_element_the_point_and_the_code_that_it_was_made_with() {
    let failure = Failure::new(123_456, 27, 7);
    assert!(failure.is_failure());
    assert_eq!(failure.item(), 123_456);
    assert_eq!(failure.point(), 27);
    assert_eq!(failure.code(), 7);
}

#[test]
fn a_failure_is_the_word_that_a_device_records() {
    let failure = Failure::new(5, 3, 1);
    assert_eq!(Failure::from_bits(failure.bits()), failure);
    assert_eq!(failure.bits(), (5 << 32) | (3 << 16) | (1 << 8));
}

#[test]
fn the_least_failure_is_that_of_the_first_element_then_the_first_point_then_the_least_code() {
    let failures = [
        Failure::new(0, 0, 1),
        Failure::new(0, 0, 2),
        Failure::new(0, 1, 1),
        Failure::new(1, 0, 1),
        Failure::new(u32::MAX - 1, u16::MAX, u8::MAX),
    ];
    failures
        .windows(2)
        .for_each(|pair| assert!(pair[0] < pair[1]));
}

#[test]
fn no_failure_is_after_every_failure_and_is_not_one() {
    assert!(!Failure::NONE.is_failure());
    assert_eq!(Failure::NONE.bits(), u64::MAX);
    assert!(Failure::new(u32::MAX - 1, u16::MAX, u8::MAX) < Failure::NONE);
    assert_eq!(
        Failure::NONE.min(Failure::new(9, 9, 9)),
        Failure::new(9, 9, 9)
    );
}

#[test]
fn the_failure_of_an_error_has_its_point_and_its_code() {
    let error = FiniteElementError::new(Some(5), ConstitutiveError::InvalidJacobian(-1.0));
    let failure = Failure::of(42, &error);
    assert_eq!(failure.item(), 42);
    assert_eq!(failure.point(), 5);
    assert_eq!(failure.code(), error.source.code());
    assert!(failure.is_failure());
}

#[test]
fn an_error_with_no_point_is_at_the_first() {
    let error = FiniteElementError::new(None, ConstitutiveError::MaximumExtensibility);
    assert_eq!(Failure::of(0, &error).point(), 0);
}

#[test]
fn an_element_that_there_cannot_be_is_the_last_that_there_can_be() {
    let error = FiniteElementError::new(None, ConstitutiveError::MaximumExtensibility);
    let failure = Failure::of(usize::MAX, &error);
    assert!(failure.is_failure());
    assert_eq!(failure.item(), (u32::MAX - 1) as usize);
}
