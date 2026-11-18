#[cfg(all(test, feature = "fem"))]
mod test;

use crate::{
    constitutive::ConstitutiveError,
    domain::block::element::{ElementError, ElementKind},
    math::assert::AssertionError,
};
use core::{
    error::Error,
    fmt::{self, Debug, Display, Formatter},
    iter::Sum,
};

/// An error of an item (an element), with the index of the item.
pub struct ItemError<E> {
    pub item: usize,
    pub source: E,
}

/// The parts of an error that are located by the layers it passes through.
pub struct Located {
    block: Option<usize>,
    item: Option<usize>,
    point: Option<usize>,
    kind: Option<&'static str>,
    source: Source,
}

/// Errors that know where in a model they occurred, as far as their layer does.
pub trait Locate {
    fn locate(self) -> Located;
}

/// What went wrong in an element model.
#[derive(PartialEq)]
pub enum Source {
    /// A constitutive model failed.
    Constitutive(ConstitutiveError),
    /// Something failed that a host-only routine, such as the applied loads, describes by text.
    Text(String),
    /// The model does not support what was asked.
    Unsupported(&'static str),
}

impl Display for Source {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Constitutive(error) => write!(f, "{error}"),
            Self::Text(text) => write!(f, "{text}"),
            Self::Unsupported(what) => write!(f, "{what}"),
        }
    }
}

impl Located {
    fn from_source(source: Source) -> Self {
        Self {
            block: None,
            item: None,
            point: None,
            kind: None,
            source,
        }
    }
}

impl Locate for ConstitutiveError {
    fn locate(self) -> Located {
        Located::from_source(Source::Constitutive(self))
    }
}

impl Locate for String {
    fn locate(self) -> Located {
        Located::from_source(Source::Text(self))
    }
}

impl Locate for Source {
    fn locate(self) -> Located {
        Located::from_source(self)
    }
}

impl<K: ElementKind> Locate for ElementError<K> {
    fn locate(self) -> Located {
        Located {
            block: None,
            item: None,
            point: self.point,
            kind: Some(K::NAME),
            source: Source::Constitutive(self.source),
        }
    }
}

impl<E: Locate> Locate for ItemError<E> {
    fn locate(self) -> Located {
        Located {
            item: Some(self.item),
            ..self.source.locate()
        }
    }
}

/// An error in the elements of a model, with where it occurred.
///
/// The error is plain data: the model is named by its type, and the block, element, and
/// integration point by their (zero-based) indices, each set by the layer that knows it.
pub struct ElementModelError {
    pub block: Option<usize>,
    pub item: Option<usize>,
    pub point: Option<usize>,
    pub kind: Option<&'static str>,
    pub model: &'static str,
    pub source: Source,
}

impl ElementModelError {
    pub fn new(model: &'static str, error: impl Locate) -> Self {
        let located = error.locate();
        Self {
            block: located.block,
            item: located.item,
            point: located.point,
            kind: located.kind,
            model,
            source: located.source,
        }
    }
    /// Places the error in a block, `offset` blocks after the first of the blocks that
    /// contain the error's block.
    pub fn in_block(mut self, offset: usize) -> Self {
        self.block = Some(self.block.unwrap_or(0) + offset);
        self
    }
}

impl<E> ItemError<E> {
    pub fn new(item: usize, source: E) -> Self {
        Self { item, source }
    }
}

impl Display for ElementModelError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.source)?;
        let kind = self.kind.unwrap_or("element");
        match (self.point, self.item, self.block) {
            (None, None, None) => {}
            (point, item, block) => {
                write!(f, "\nAt")?;
                if let Some(point) = point {
                    write!(f, " integration point {point}")?;
                }
                if let Some(item) = item {
                    write!(f, " of {kind} {item}")?;
                }
                if let Some(block) = block {
                    write!(f, " in block {block}")?;
                }
                write!(f, ".")?;
            }
        }
        write!(f, "\nIn {}.", self.model)
    }
}

impl Debug for ElementModelError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "\n{self}\n")
    }
}

impl Error for ElementModelError {}

impl From<ElementModelError> for String {
    fn from(error: ElementModelError) -> Self {
        error.to_string()
    }
}

impl From<ElementModelError> for AssertionError {
    fn from(error: ElementModelError) -> Self {
        Self {
            message: error.to_string(),
        }
    }
}

/// Collect or sum results over the items of a block, recording the index of the failing item.
pub trait AtItems<T, E>
where
    Self: Iterator<Item = Result<T, E>> + Sized,
{
    fn sum_at_items<S: Sum<T>>(self) -> Result<S, ItemError<E>> {
        self.enumerate()
            .map(|(item, result)| result.map_err(|source| ItemError { item, source }))
            .sum()
    }
    fn collect_at_items<C: FromIterator<T>>(self) -> Result<C, ItemError<E>> {
        self.enumerate()
            .map(|(item, result)| result.map_err(|source| ItemError { item, source }))
            .collect()
    }
}

impl<I, T, E> AtItems<T, E> for I where I: Iterator<Item = Result<T, E>> {}

/// Run a fallible function over the items of a block, recording the index of the failing item.
pub trait ForEachAtItems
where
    Self: Iterator + Sized,
{
    fn try_for_each_at_items<E>(
        self,
        mut function: impl FnMut(Self::Item) -> Result<(), E>,
    ) -> Result<(), ItemError<E>> {
        self.enumerate()
            .try_for_each(|(item, x)| function(x).map_err(|source| ItemError { item, source }))
    }
}

impl<I: Iterator> ForEachAtItems for I {}
