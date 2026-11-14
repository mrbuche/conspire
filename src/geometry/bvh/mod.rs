mod base;
mod from;
mod node;
mod primitive;
mod ray;

#[cfg(test)]
pub(crate) use base::closest_point_on_triangle;

use crate::{geometry::bvh::node::Nodes, math::Quantity, units::Length};

/// Bounding volume hierarchy over triangle primitives.
pub struct BoundingVolumeHierarchy<const D: usize> {
    items: Vec<usize>,
    nodes: Nodes<D>,
}

/// Ray intersection with a primitive.
#[derive(Clone, Debug, PartialEq)]
pub struct Hit {
    distance: Quantity<Length>,
    index: usize,
}

impl Hit {
    /// Distance along the ray to the intersection.
    pub fn distance(&self) -> Quantity<Length> {
        self.distance
    }
    /// Index of the intersected primitive.
    pub fn index(&self) -> usize {
        self.index
    }
}
