use super::{GpuError, Imposed, ImposedLoad, Schedule};

fn invalid(result: Result<impl Sized, GpuError>) -> String {
    match result {
        Err(GpuError::Invalid(message)) => message,
        Err(error) => panic!("expected an invalid run: {error}"),
        Ok(_) => panic!("expected an invalid run"),
    }
}

#[test]
fn the_degrees_of_freedom_that_are_held_or_moved_have_no_acceleration() -> Result<(), GpuError> {
    let imposed = Imposed {
        fixed: vec![0, 4],
        moved: vec![5, 4],
        ..Imposed::default()
    };
    assert_eq!(imposed.held_flags(2)?, [1, 0, 0, 0, 1, 1]);
    assert_eq!(Imposed::default().held_flags(1)?, [0, 0, 0]);
    assert_eq!(Imposed::default().held_flags(0)?, Vec::<u32>::new());
    Ok(())
}

#[test]
fn a_degree_of_freedom_that_the_model_does_not_have_is_an_error_that_says_which() {
    for imposed in [
        Imposed {
            fixed: vec![2, 6],
            ..Imposed::default()
        },
        Imposed {
            moved: vec![6],
            ..Imposed::default()
        },
    ] {
        let message = invalid(imposed.held_flags(2));
        assert!(message.contains("degree of freedom 6"), "{message}");
        assert!(message.contains("not in the model"), "{message}");
        assert!(imposed.check(2).is_err());
    }
}

#[test]
fn a_load_has_a_value_for_each_degree_of_freedom_or_a_weight_for_each_node() -> Result<(), GpuError>
{
    let loads = |constant: usize, weights: usize| Imposed {
        loads: vec![
            ImposedLoad::Constant(vec![0.0; constant]),
            ImposedLoad::Weighted(vec![0.0; weights]),
        ],
        ..Imposed::default()
    };
    loads(6, 2).check(2)?;
    let message = invalid(loads(5, 2).check(2));
    assert!(
        message.contains("5 values") && message.contains("6 degrees"),
        "{message}"
    );
    let message = invalid(loads(6, 3).check(2));
    assert!(
        message.contains("3 weights") && message.contains("2 nodes"),
        "{message}"
    );
    Ok(())
}

#[test]
fn the_loads_that_are_a_function_of_the_step_are_the_weighted_ones() {
    let imposed = Imposed {
        loads: vec![
            ImposedLoad::Constant(Vec::new()),
            ImposedLoad::Weighted(Vec::new()),
            ImposedLoad::Constant(Vec::new()),
            ImposedLoad::Weighted(Vec::new()),
        ],
        ..Imposed::default()
    };
    assert_eq!(imposed.weighted(), 2);
    assert_eq!(Imposed::default().weighted(), 0);
}

#[test]
fn a_table_holds_as_many_rows_as_fit_in_a_few_million_values_and_a_thousand_at_most() {
    let moved = |count| Imposed {
        moved: vec![0; count],
        ..Imposed::default()
    };
    assert_eq!(moved(0).rows(), 1024);
    assert_eq!(moved(300).rows(), 1024);
    assert_eq!(moved(100_000).rows(), (1 << 22) / 100_000);
    assert_eq!(moved(10_000_000).rows(), 1);
}

#[test]
fn a_schedule_has_targets_for_coordinates_that_are_moved_and_components_for_loads_of_time() {
    let imposed = Imposed {
        moved: vec![0],
        loads: vec![ImposedLoad::Weighted(vec![1.0])],
        ..Imposed::default()
    };
    let targets = |_| vec![0.0];
    let components = |_| [0.0; 3];
    let schedule = |targets, components| Schedule {
        targets,
        components,
    };
    assert!(
        schedule(Some(&targets), vec![&components])
            .check(&imposed)
            .is_ok()
    );
    let message = invalid(schedule(None, vec![&components]).check(&imposed));
    assert!(message.contains("targets"), "{message}");
    let message = invalid(schedule(Some(&targets), Vec::new()).check(&imposed));
    assert!(message.contains("components"), "{message}");
    assert!(Schedule::default().check(&Imposed::default()).is_ok());
}

#[test]
fn the_targets_of_the_rows_are_in_the_order_of_the_rows() -> Result<(), GpuError> {
    let targets = |row: usize| vec![row as f64, 10.0 + row as f64];
    let schedule = Schedule {
        targets: Some(&targets),
        ..Schedule::default()
    };
    assert_eq!(schedule.target_rows(2, 3, 2)?, [3.0, 13.0, 4.0, 14.0]);
    assert!(schedule.target_rows(0, 3, 2)?.is_empty());
    assert!(Schedule::default().target_rows(0, 0, 5)?.is_empty());
    let message = invalid(schedule.target_rows(3, 0, 1));
    assert!(
        message.contains("2 targets") && message.contains("3 coordinates"),
        "{message}"
    );
    Ok(())
}

#[test]
fn the_components_of_a_load_are_three_to_a_row_in_the_order_of_the_rows() {
    let first = |row: usize| [row as f64, 0.0, 1.0];
    let second = |row: usize| [-(row as f64), 5.0, 2.0];
    let schedule = Schedule {
        components: vec![&first, &second],
        ..Schedule::default()
    };
    assert_eq!(
        schedule.component_rows(0, 2, 2),
        [2.0, 0.0, 1.0, 3.0, 0.0, 1.0]
    );
    assert_eq!(
        schedule.component_rows(1, 0, 2),
        [0.0, 5.0, 2.0, -1.0, 5.0, 2.0]
    );
}
