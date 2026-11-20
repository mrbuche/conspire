use super::FiniteElementError;
use crate::constitutive::ConstitutiveError;

#[test]
fn an_error_at_an_integration_point_says_where_in_the_finite_element() {
    let error = FiniteElementError::new(Some(3), ConstitutiveError::MaximumExtensibility);
    assert!(
        error
            .to_string()
            .ends_with("At integration point 3 of the finite element.")
    );
}

#[test]
fn an_error_with_no_integration_point_says_that_it_is_in_the_finite_element() {
    let error = FiniteElementError::new(None, ConstitutiveError::MaximumExtensibility);
    assert!(error.to_string().ends_with("In the finite element."));
}
