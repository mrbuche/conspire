use super::{CudssSolver, Device, GpuError};
use std::env::var_os;

fn skip<T>(why: &str) -> Option<T> {
    assert!(
        var_os("CONSPIRE_REQUIRE_GPU").is_none(),
        "CONSPIRE_REQUIRE_GPU is set and {why}"
    );
    eprintln!("skipped: {why}");
    None
}

fn solver(row_starts: &[i32], columns: &[i32], symmetric: bool) -> Option<CudssSolver> {
    let device = match Device::new(0) {
        Ok(device) => device,
        Err(GpuError::Unavailable(message)) => {
            return skip(&format!("there is no device: {message}"));
        }
        Err(error) => panic!("{error}"),
    };
    match CudssSolver::new(&device, row_starts, columns, symmetric) {
        Ok(solver) => Some(solver),
        Err(GpuError::Unavailable(message)) => skip(&format!("there is no cuDSS: {message}")),
        Err(error) => panic!("{error}"),
    }
}

fn all_near(found: &[f64], expected: f64) {
    found.iter().for_each(|value| {
        assert!(
            (value - expected).abs() < 1e-12,
            "{found:?} is not {expected}"
        )
    });
}

#[test]
fn solves_a_general_system_of_three_and_refactors() {
    let (row_starts, columns) = ([0, 2, 5, 7], [0, 1, 0, 1, 2, 1, 2]);
    let values = [4.0, 1.0, 2.0, 5.0, 1.0, 1.0, 3.0];
    let Some(mut solver) = solver(&row_starts, &columns, false) else {
        return;
    };
    let b = [5.0, 8.0, 4.0];
    all_near(&solver.solve(&values, &b).unwrap(), 1.0);
    let doubled = values.map(|value| 2.0 * value);
    all_near(&solver.solve(&doubled, &b).unwrap(), 0.5);
}

#[test]
fn solves_a_symmetric_system_from_its_upper_triangle() {
    let (row_starts, columns) = ([0, 2, 4, 5], [0, 1, 1, 2, 2]);
    let values = [4.0, 1.0, 5.0, 1.0, 3.0];
    let Some(mut solver) = solver(&row_starts, &columns, true) else {
        return;
    };
    all_near(&solver.solve(&values, &[5.0, 7.0, 4.0]).unwrap(), 1.0);
}
