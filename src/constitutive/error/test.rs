use super::{ConstitutiveError, IntegratorFailure, SolverFailure, Unavailable};
use crate::math::{
    SquareMatrixError, TensorError, assert::AssertionError, integrate::IntegrationError,
    optimize::OptimizationError,
};
use core::error::Error;

fn one_of_each() -> [ConstitutiveError; 10] {
    [
        ConstitutiveError::InvalidJacobian(-0.5),
        ConstitutiveError::MaximumExtensibility,
        ConstitutiveError::Unavailable(Unavailable::TangentStiffnessWrtDeformationGradient),
        ConstitutiveError::Tensor(TensorError::NotPositiveDefinite),
        ConstitutiveError::SquareMatrix(SquareMatrixError::Singular),
        ConstitutiveError::Solver(SolverFailure::Other),
        ConstitutiveError::Integrator(IntegratorFailure::Other),
        ConstitutiveError::ReturnMapDidNotConverge(12),
        ConstitutiveError::InvalidTimes,
        ConstitutiveError::StepBelowFloor,
    ]
}

#[test]
fn every_kind_of_error_has_a_code_of_its_own() {
    let mut codes = one_of_each().map(|error| error.code()).to_vec();
    codes.sort_unstable();
    assert_eq!(codes, (1..=10).collect::<Vec<u8>>());
}

#[test]
fn the_message_of_an_error_says_what_happened() {
    let messages = one_of_each().map(|error| error.to_string());
    assert!(messages.iter().all(|message| !message.is_empty()));
    assert!(messages[0].contains("Invalid Jacobian: -5.000000e-1."));
    assert!(messages[7].contains("12 steps"));
    assert!(
        ConstitutiveError::Solver(SolverFailure::MaximumStepsReached(7))
            .to_string()
            .contains("(7)")
    );
    assert!(
        ConstitutiveError::Unavailable(Unavailable::StiffnessesAtNonzeroVelocity)
            .to_string()
            .contains("zero velocities")
    );
}

#[test]
fn the_debug_of_an_error_is_its_message_between_newlines() {
    for error in one_of_each() {
        assert_eq!(format!("{error:?}"), format!("\n{error}\n"));
    }
}

#[test]
fn errors_are_equal_when_they_are_of_a_kind_and_carry_the_same() {
    assert_eq!(
        ConstitutiveError::InvalidJacobian(-0.5),
        ConstitutiveError::InvalidJacobian(-0.5)
    );
    assert_ne!(
        ConstitutiveError::InvalidJacobian(-0.5),
        ConstitutiveError::InvalidJacobian(-0.25)
    );
    assert_ne!(
        ConstitutiveError::InvalidTimes,
        ConstitutiveError::StepBelowFloor
    );
    assert_ne!(
        ConstitutiveError::Solver(SolverFailure::SingularMatrix),
        ConstitutiveError::Solver(SolverFailure::UnsymmetricMatrix)
    );
    for (index, error) in one_of_each().iter().enumerate() {
        for (other_index, other) in one_of_each().iter().enumerate() {
            assert_eq!(error == other, index == other_index);
        }
    }
}

#[test]
fn a_solver_that_failed_keeps_how_it_failed_and_not_what_it_said() {
    let failure = |error| match ConstitutiveError::from(error) {
        ConstitutiveError::Solver(failure) => failure,
        other => panic!("expected a solver failure, found {other:?}"),
    };
    assert_eq!(
        failure(OptimizationError::MaximumStepsReached(
            5,
            "a long text".to_string()
        )),
        SolverFailure::MaximumStepsReached(5)
    );
    assert_eq!(
        failure(OptimizationError::NotMinimum(
            "a".to_string(),
            "b".to_string()
        )),
        SolverFailure::NotMinimum
    );
    assert_eq!(
        failure(OptimizationError::SingularMatrix),
        SolverFailure::SingularMatrix
    );
    assert_eq!(
        failure(OptimizationError::UnsymmetricMatrix),
        SolverFailure::UnsymmetricMatrix
    );
    assert_eq!(
        failure(OptimizationError::Intermediate("a".to_string())),
        SolverFailure::Other
    );
    assert_eq!(
        failure(OptimizationError::Upstream(
            "a".to_string(),
            "b".to_string()
        )),
        SolverFailure::Other
    );
}

#[test]
fn an_integrator_that_failed_keeps_how_it_failed_and_not_what_it_said() {
    let failure = |error| match ConstitutiveError::from(error) {
        ConstitutiveError::Integrator(failure) => failure,
        other => panic!("expected an integrator failure, found {other:?}"),
    };
    assert_eq!(
        failure(IntegrationError::MinimumStepSizeReached(
            1e-3,
            "an integrator".to_string()
        )),
        IntegratorFailure::MinimumStepSizeReached(1e-3)
    );
    assert_eq!(
        failure(IntegrationError::MinimumStepSizeUpstream(
            1e-3,
            "an error".to_string(),
            "an integrator".to_string()
        )),
        IntegratorFailure::MinimumStepSizeReached(1e-3)
    );
    assert_eq!(
        failure(IntegrationError::TimeStepNotSet(
            0.0,
            1.0,
            "an integrator".to_string()
        )),
        IntegratorFailure::TimeStepNotSet(0.0, 1.0)
    );
    assert_eq!(
        failure(IntegrationError::UnstableTimeStep(
            2.0,
            1.0,
            "an integrator".to_string()
        )),
        IntegratorFailure::UnstableTimeStep(2.0, 1.0)
    );
    assert_eq!(
        failure(IntegrationError::InvalidSafetyFactor(2.0)),
        IntegratorFailure::InvalidSafetyFactor(2.0)
    );
    assert_eq!(
        failure(IntegrationError::LengthTimeLessThanTwo),
        IntegratorFailure::LengthTimeLessThanTwo
    );
    assert_eq!(
        failure(IntegrationError::Intermediate("a".to_string())),
        IntegratorFailure::Other
    );
    assert_eq!(
        failure(IntegrationError::Upstream("a".to_string(), "b".to_string())),
        IntegratorFailure::Other
    );
}

#[test]
fn the_errors_of_tensors_and_matrices_are_errors_of_the_model() {
    assert_eq!(
        ConstitutiveError::from(TensorError::NotPositiveDefinite),
        ConstitutiveError::Tensor(TensorError::NotPositiveDefinite)
    );
    assert_eq!(
        ConstitutiveError::from(SquareMatrixError::Singular),
        ConstitutiveError::SquareMatrix(SquareMatrixError::Singular)
    );
}

#[test]
fn an_error_becomes_its_message_as_text_and_as_a_failed_assertion_and_a_failed_minimization() {
    for error in one_of_each() {
        let message = error.to_string();
        assert_eq!(AssertionError::from(error).message, message);
    }
    let error = ConstitutiveError::MaximumExtensibility;
    let message = error.to_string();
    assert_eq!(String::from(error), message);
    let error = ConstitutiveError::StepBelowFloor;
    let message = error.to_string();
    match OptimizationError::from(error) {
        OptimizationError::Intermediate(text) => assert_eq!(text, message),
        other => panic!("expected an intermediate error, found {other:?}"),
    }
}

#[test]
fn an_error_of_a_model_is_an_error() {
    fn is_an_error(_: &dyn Error) {}
    is_an_error(&ConstitutiveError::InvalidTimes);
}
