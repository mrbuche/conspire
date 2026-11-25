use super::{GpuError, Imposed};

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
    }
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
