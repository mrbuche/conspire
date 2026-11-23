use super::{GpuError, held_flags};

#[test]
fn the_held_degrees_of_freedom_are_the_ones_that_are_flagged() -> Result<(), GpuError> {
    assert_eq!(held_flags(6, &[0, 4, 5])?, [1, 0, 0, 0, 1, 1]);
    assert_eq!(held_flags(3, &[])?, [0, 0, 0]);
    assert_eq!(held_flags(0, &[])?, Vec::<u32>::new());
    Ok(())
}

#[test]
fn a_degree_of_freedom_that_is_held_twice_is_held() -> Result<(), GpuError> {
    assert_eq!(held_flags(3, &[1, 1])?, [0, 1, 0]);
    Ok(())
}

#[test]
fn a_degree_of_freedom_that_the_model_does_not_have_is_an_error_that_says_which() {
    let Err(error) = held_flags(6, &[2, 6]) else {
        panic!("a degree of freedom that is not in the model is held");
    };
    assert!(matches!(error, GpuError::Invalid(_)));
    let message = error.to_string();
    assert!(message.contains("degree of freedom 6"), "{message}");
    assert!(message.contains("6 of them"), "{message}");
}
