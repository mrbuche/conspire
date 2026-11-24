use super::Execution;

#[test]
fn serial_is_the_default_and_is_one_thread() {
    assert_eq!(Execution::default(), Execution::Serial);
    assert_eq!(Execution::Serial.threads(), 1);
}

#[test]
fn threads_are_the_number_given() {
    assert_eq!(Execution::Threads(1).threads(), 1);
    assert_eq!(Execution::Threads(7).threads(), 7);
}

#[test]
fn zero_threads_is_as_many_as_the_machine_offers() {
    assert!(Execution::Threads(0).threads() >= 1);
}
