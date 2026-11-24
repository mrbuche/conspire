#[cfg(test)]
mod test;

mod record;

use crate::domain::block::element::{ElementError, ElementKind};

pub use record::ErrorRecord;

pub struct FiniteElementKind;

impl ElementKind for FiniteElementKind {
    const NAME: &'static str = "finite element";
}

pub type FiniteElementError = ElementError<FiniteElementKind>;
