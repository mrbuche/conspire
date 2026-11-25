use super::Schedule;
use crate::{gpu::GpuError, math::Quantity, units::Time};

fn invalid(result: Result<impl Sized, GpuError>) -> String {
    match result {
        Err(GpuError::Invalid(message)) => message,
        Err(error) => panic!("expected an invalid run: {error}"),
        Ok(_) => panic!("expected an invalid run"),
    }
}

struct Steps {
    times: bool,
    targets: bool,
}

impl Schedule for Steps {
    fn time(&self, row: usize) -> Option<Quantity<Time>> {
        self.times.then(|| Time::seconds(0.5 * row as f64))
    }
    fn targets(&self, row: usize) -> Option<Vec<f64>> {
        self.targets.then(|| vec![row as f64, 10.0 + row as f64])
    }
}

#[test]
fn a_schedule_has_targets_for_coordinates_that_are_moved_and_times_for_loads_of_time() {
    let both = Steps {
        times: true,
        targets: true,
    };
    assert!(both.check(true, 1).is_ok());
    let no_targets = Steps {
        targets: false,
        ..both
    };
    let message = invalid(no_targets.check(true, 1));
    assert!(message.contains("targets"), "{message}");
    let no_times = Steps {
        times: false,
        targets: true,
    };
    let message = invalid(no_times.check(true, 3));
    assert!(
        message.contains("times") && message.contains("3 loads"),
        "{message}"
    );
}

#[test]
fn a_schedule_gives_only_what_a_run_needs_of_it() {
    assert!(().check(false, 0).is_ok());
    let no_targets = Steps {
        times: true,
        targets: false,
    };
    assert!(no_targets.check(false, 2).is_ok());
    let no_times = Steps {
        times: false,
        targets: true,
    };
    assert!(no_times.check(true, 0).is_ok());
    assert!(invalid(().check(true, 0)).contains("targets"));
    assert!(invalid(().check(false, 1)).contains("times"));
}

#[test]
fn the_times_of_the_rows_are_in_the_order_of_the_rows() -> Result<(), GpuError> {
    let schedule = Steps {
        times: true,
        targets: false,
    };
    assert_eq!(
        schedule.times(2, 3)?,
        [Time::seconds(1.0), Time::seconds(1.5), Time::seconds(2.0)]
    );
    assert!(schedule.times(2, 0)?.is_empty());
    let message = invalid(().times(0, 1));
    assert!(message.contains("no time for the row 0"), "{message}");
    Ok(())
}

#[test]
fn the_targets_of_the_rows_are_in_the_order_of_the_rows() -> Result<(), GpuError> {
    let schedule = Steps {
        times: false,
        targets: true,
    };
    assert_eq!(schedule.target_rows(2, 3, 2)?, [3.0, 13.0, 4.0, 14.0]);
    assert!(schedule.target_rows(0, 3, 2)?.is_empty());
    assert!(().target_rows(0, 0, 5)?.is_empty());
    let message = invalid(schedule.target_rows(3, 0, 1));
    assert!(
        message.contains("2 targets") && message.contains("3 coordinates"),
        "{message}"
    );
    let message = invalid(().target_rows(1, 4, 1));
    assert!(message.contains("no targets for the row 4"), "{message}");
    Ok(())
}
