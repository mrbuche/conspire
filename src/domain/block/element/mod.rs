pub(crate) mod solid;
#[cfg(test)]
pub(crate) mod test;

use crate::{
    constitutive::ConstitutiveError, domain::block::add_node_neighbors,
    math::assert::AssertionError,
};
use core::{
    fmt::{self, Debug, Display, Formatter},
    iter::Sum,
    marker::PhantomData,
};

pub trait Elements
where
    Self: Debug,
{
    fn node_neighbors(&self, neighbors: &mut [Vec<usize>]);
    /// The number of blocks, being one for a single block.
    fn number_of_blocks(&self) -> usize {
        1
    }
}

/// The structure of a block: the items it is made of, and the nodes of each.
///
/// An item is an element, or whatever else contributes to a set of nodes. Every [`Items`] is
/// [`Elements`], and the traits that assemble over the items extend this one with their kernels.
pub trait Items
where
    Self: Debug,
{
    /// The number of items.
    fn number_of_items(&self) -> usize;
    /// The nodes of an item.
    fn nodes_of(&self, item: usize) -> &[usize];
}

impl<T> Elements for T
where
    T: Items,
{
    fn node_neighbors(&self, neighbors: &mut [Vec<usize>]) {
        add_node_neighbors(
            (0..self.number_of_items()).map(|item| self.nodes_of(item)),
            neighbors,
        )
    }
}

pub trait ElementKind {
    const NAME: &'static str;
}

/// An error in an element, with the integration point where it occurred, if known.
///
/// The error is plain data. The element and block that contain it are added by the
/// layers that know them.
pub struct ElementError<K> {
    pub point: Option<usize>,
    pub source: ConstitutiveError,
    kind: PhantomData<K>,
}

impl<K> ElementError<K> {
    pub fn new(point: Option<usize>, source: ConstitutiveError) -> Self {
        Self {
            point,
            source,
            kind: PhantomData,
        }
    }
}

impl<K> PartialEq for ElementError<K> {
    fn eq(&self, other: &Self) -> bool {
        self.point == other.point && self.source == other.source
    }
}

impl<K> From<ConstitutiveError> for ElementError<K> {
    fn from(source: ConstitutiveError) -> Self {
        Self::new(None, source)
    }
}

impl<K> From<PointError> for ElementError<K> {
    fn from(error: PointError) -> Self {
        Self::new(Some(error.point), error.source)
    }
}

impl<K: ElementKind> From<ElementError<K>> for AssertionError {
    fn from(error: ElementError<K>) -> Self {
        Self {
            message: error.to_string(),
        }
    }
}

impl<K: ElementKind> Debug for ElementError<K> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "\n{self}\n")
    }
}

impl<K: ElementKind> Display for ElementError<K> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self.point {
            Some(point) => write!(
                f,
                "{}\nAt integration point {point} of the {}.",
                self.source,
                K::NAME
            ),
            None => write!(f, "{}\nIn the {}.", self.source, K::NAME),
        }
    }
}

impl<K: ElementKind> core::error::Error for ElementError<K> {}

/// An error at an integration point, before the element is known.
pub struct PointError {
    pub point: usize,
    pub source: ConstitutiveError,
}

/// Collect or sum results over the integration points, recording the point of a failure.
#[cfg_attr(not(any(feature = "fem", feature = "vem")), allow(dead_code))]
pub(crate) trait AtPoints<T>
where
    Self: Iterator<Item = Result<T, ConstitutiveError>> + Sized,
{
    fn collect_at_points<C: FromIterator<T>>(self) -> Result<C, PointError> {
        self.enumerate()
            .map(|(point, result)| result.map_err(|source| PointError { point, source }))
            .collect()
    }
    fn sum_at_points<S: Sum<T>>(self) -> Result<S, PointError> {
        self.enumerate()
            .map(|(point, result)| result.map_err(|source| PointError { point, source }))
            .sum()
    }
}

impl<I, T> AtPoints<T> for I where I: Iterator<Item = Result<T, ConstitutiveError>> {}
