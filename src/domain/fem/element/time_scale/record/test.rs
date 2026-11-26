use super::UnstableRecord;

#[test]
fn a_record_keeps_its_step_and_element() {
    let record = UnstableRecord::of(7, 123_456);
    assert!(record.is_unstable());
    assert_eq!((record.step(), record.item()), (7, 123_456));
    assert_eq!(UnstableRecord::from_bits(record.bits()), record);
}

#[test]
fn the_least_record_is_that_of_the_first_step_and_then_of_the_first_element() {
    assert!(UnstableRecord::of(1, 900) < UnstableRecord::of(2, 0));
    assert!(UnstableRecord::of(2, 3) < UnstableRecord::of(2, 4));
}

#[test]
fn every_record_is_less_than_none() {
    let record = UnstableRecord::of(u32::MAX, usize::MAX);
    assert!(record < UnstableRecord::NONE);
    assert!(record.is_unstable());
    assert!(!UnstableRecord::NONE.is_unstable());
}
