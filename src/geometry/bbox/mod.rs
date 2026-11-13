#[cfg(test)]
pub mod test;

mod base;
mod from;
mod unite;

use crate::geometry::Coordinate;

/// Axis-aligned bounding box in `D` dimensions.
#[derive(Clone, Debug, PartialEq)]
pub struct BoundingBox<const D: usize> {
    minimum: Coordinate<D>,
    maximum: Coordinate<D>,
}

/// Bounding boxes.
pub type BoundingBoxes<const D: usize> = Vec<BoundingBox<D>>;

/// Combine two values into one enclosing both.
pub trait Unite<T> {
    /// Combined type.
    type Output;
    /// Smallest enclosing combination of `self` and `other`.
    fn unite(self, other: T) -> Self::Output;
}
