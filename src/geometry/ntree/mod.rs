mod balance;
mod defeature;
mod deref;
mod from;
mod index;
mod into;
mod leaves;
pub(crate) mod node;
mod pair;
mod prune;
mod read;
pub(crate) mod rescale;
pub(crate) mod sizing;
pub(crate) mod subdivide;
mod write;

pub use crate::geometry::mesh::Dualization;
pub use crate::geometry::ntree::{
    balance::{Balance, Balancing},
    node::Nodes,
    pair::Pairing,
    read::Input,
    rescale::Rescaling,
    sizing::{Sizing, curvature::CurvatureSizing},
    write::Output,
};

/// Orthotree in `D` dimensions.
pub struct Orthotree<const D: usize, const L: usize, const M: usize, const N: usize, T, U, V = ()> {
    pub(crate) balanced: Balancing,
    pub(crate) nodes: Nodes<D, M, N, T, U, V>,
    pub(crate) paired: Pairing,
    pub(crate) rescale: Rescaling<D>,
}

/// Binary tree (1D).
pub type BinaryTree<T, U, V = ()> = Orthotree<1, 1, 2, 2, T, U, V>;
/// Quadtree (2D).
pub type Quadtree<T, U, V = ()> = Orthotree<2, 2, 4, 4, T, U, V>;
/// Octree (3D).
pub type Octree<T, U, V = ()> = Orthotree<3, 4, 6, 8, T, U, V>;
/// Hexadecatree (4D).
pub type Hexadecatree<T, U, V = ()> = Orthotree<4, 8, 8, 16, T, U, V>;
