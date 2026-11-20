use super::Resources;

#[test]
fn the_occupancy_is_the_fraction_of_the_threads_of_a_multiprocessor_that_the_blocks_have() {
    let resources = Resources {
        registers: 255,
        local_bytes: 0,
        threads_per_block: 128,
        blocks_per_multiprocessor: 2,
        threads_per_multiprocessor: 2048,
    };
    assert_eq!(resources.occupancy(), 0.125);
    assert_eq!(
        Resources {
            blocks_per_multiprocessor: 16,
            ..resources
        }
        .occupancy(),
        1.0
    );
}

#[test]
fn a_device_that_reports_no_threads_does_not_divide_by_zero() {
    let resources = Resources {
        registers: 32,
        local_bytes: 0,
        threads_per_block: 128,
        blocks_per_multiprocessor: 1,
        threads_per_multiprocessor: 0,
    };
    assert!(resources.occupancy().is_finite());
}
