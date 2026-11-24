use super::ErrorRecord;
use crate::{constitutive::ConstitutiveError, fem::element::FiniteElementError};

#[test]
fn a_record_is_the_element_the_point_and_the_code_that_it_was_made_with() {
    let record = ErrorRecord::new(123_456, 27, 7);
    assert!(record.is_error());
    assert_eq!(record.item(), 123_456);
    assert_eq!(record.point(), 27);
    assert_eq!(record.code(), 7);
}

#[test]
fn a_record_is_the_word_that_a_device_records() {
    let record = ErrorRecord::new(5, 3, 1);
    assert_eq!(ErrorRecord::from_bits(record.bits()), record);
    assert_eq!(record.bits(), (5 << 32) | (3 << 16) | (1 << 8));
}

#[test]
fn the_least_record_is_that_of_the_first_element_then_the_first_point_then_the_least_code() {
    let records = [
        ErrorRecord::new(0, 0, 1),
        ErrorRecord::new(0, 0, 2),
        ErrorRecord::new(0, 1, 1),
        ErrorRecord::new(1, 0, 1),
        ErrorRecord::new(u32::MAX - 1, u16::MAX, u8::MAX),
    ];
    records
        .windows(2)
        .for_each(|pair| assert!(pair[0] < pair[1]));
}

#[test]
fn no_error_is_after_every_record_and_is_not_one() {
    assert!(!ErrorRecord::NONE.is_error());
    assert_eq!(ErrorRecord::NONE.bits(), u64::MAX);
    assert!(ErrorRecord::new(u32::MAX - 1, u16::MAX, u8::MAX) < ErrorRecord::NONE);
    assert_eq!(
        ErrorRecord::NONE.min(ErrorRecord::new(9, 9, 9)),
        ErrorRecord::new(9, 9, 9)
    );
}

#[test]
fn a_word_with_every_bit_set_is_no_error() {
    let word = u64::from(u32::MAX) | (u64::from(u32::MAX) << 32);
    assert_eq!(ErrorRecord::from_bits(word), ErrorRecord::NONE);
    assert!(!ErrorRecord::from_bits(word).is_error());
}

#[test]
fn the_record_of_an_error_has_its_point_and_its_code() {
    let error = FiniteElementError::new(Some(5), ConstitutiveError::InvalidJacobian(-1.0));
    let record = ErrorRecord::of(42, &error);
    assert_eq!(record.item(), 42);
    assert_eq!(record.point(), 5);
    assert_eq!(record.code(), error.source.code());
    assert!(record.is_error());
}

#[test]
fn an_error_with_no_point_is_at_the_first() {
    let error = FiniteElementError::new(None, ConstitutiveError::MaximumExtensibility);
    assert_eq!(ErrorRecord::of(0, &error).point(), 0);
}

#[test]
fn an_element_that_there_cannot_be_is_the_last_that_there_can_be() {
    let error = FiniteElementError::new(None, ConstitutiveError::MaximumExtensibility);
    let record = ErrorRecord::of(usize::MAX, &error);
    assert!(record.is_error());
    assert_eq!(record.item(), (u32::MAX - 1) as usize);
}
