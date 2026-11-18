use super::{AtItems, ElementModelError, ItemError};
use crate::{constitutive::ConstitutiveError, fem::element::FiniteElementError};

#[test]
fn the_error_names_the_point_element_and_block() {
    let error = ElementModelError::new(
        "a model",
        ItemError::new(
            4,
            FiniteElementError::new(Some(2), ConstitutiveError::MaximumExtensibility),
        ),
    )
    .in_block(1);
    let message = error.to_string();
    assert!(message.contains("Maximum extensibility reached."));
    assert!(message.contains("At integration point 2 of finite element 4 in block 1."));
    assert!(message.contains("In a model."));
}

#[test]
fn a_block_is_the_first_unless_placed() {
    let error = ElementModelError::new("a model", ConstitutiveError::MaximumExtensibility);
    assert_eq!(error.block, None);
    assert_eq!(error.in_block(0).block, Some(0));
}

#[test]
fn the_failing_item_is_the_index_among_the_items() {
    let results = [
        Ok(1.0),
        Ok(2.0),
        Err(ConstitutiveError::MaximumExtensibility),
    ];
    let error = results
        .into_iter()
        .sum_at_items::<f64>()
        .expect_err("the third item fails");
    assert_eq!(error.item, 2);
}
