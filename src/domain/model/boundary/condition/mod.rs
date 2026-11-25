#[cfg(test)]
mod test;

mod fixed;
mod prescribed;
mod rows;
mod timed;

pub use fixed::Fixed;
pub use prescribed::Prescribed;
pub use rows::Rows;
pub use timed::Timed;

use crate::{
    domain::boundary::{Component, Layout, Prescribable},
    geometry::mesh::NodeSet,
    math::{
        Quantity,
        optimize::{EqualityConstraint, EqualityConstraints, Prescription},
    },
    units::Time,
};
use core::marker::PhantomData;

/// How degrees of freedom are held.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Enforcement {
    /// They are eliminated from the system.
    Elimination,
    /// They are held by Lagrange multipliers.
    Multipliers,
}

enum Target<'m> {
    Set(&'m NodeSet),
    Nodes(Vec<usize>),
}

impl Target<'_> {
    fn indices(&self, stride: usize, offset: usize) -> Vec<usize> {
        self.dofs(stride, offset).collect()
    }
    /// The degrees of freedom of the nodes, in the order of the nodes, without a list of them.
    fn dofs(&self, stride: usize, offset: usize) -> impl Iterator<Item = usize> + '_ {
        let nodes = match self {
            Self::Set(set) => set.nodes(),
            Self::Nodes(nodes) => nodes,
        };
        nodes.iter().map(move |&node| stride * node + offset)
    }
}

fn each_prescribed(constraint: EqualityConstraint, each: &mut impl FnMut(usize, Prescription)) {
    if let EqualityConstraint::Prescribed(entries) | EqualityConstraint::Multipliers(entries) =
        constraint
    {
        entries
            .into_iter()
            .filter(|&(_, prescription)| prescription != Prescription::Initial)
            .for_each(|(index, prescription)| each(index, prescription));
    }
}

fn held(entries: Vec<(usize, Prescription)>, enforcement: Enforcement) -> EqualityConstraint {
    match enforcement {
        Enforcement::Multipliers => EqualityConstraint::Multipliers(entries),
        Enforcement::Elimination
            if entries
                .iter()
                .all(|&(_, prescription)| prescription == Prescription::Initial) =>
        {
            let mut indices: Vec<usize> = entries.into_iter().map(|(index, _)| index).collect();
            indices.sort_unstable();
            indices.dedup();
            EqualityConstraint::Fixed(indices)
        }
        Enforcement::Elimination => EqualityConstraint::Prescribed(entries),
    }
}

/// A condition on the degrees of freedom of a layout, which may be a function of time.
pub trait Condition<L> {
    /// The constraint holding the degrees of freedom at a time.
    fn constraint_at(&self, time: Quantity<Time>) -> EqualityConstraint;
    /// Gives `each` degree of freedom that is held at a time at a value other than the one that it
    /// starts at, with what that is, in the order of the constraint.
    ///
    /// A run asks this at every step, so a condition that can say it without making its
    /// constraint is better to.
    fn prescribed_at(&self, time: Quantity<Time>, each: &mut impl FnMut(usize, Prescription)) {
        each_prescribed(self.constraint_at(time), each)
    }
}

/// A condition on the degrees of freedom of a layout that is not a function of time.
pub trait Static<L>: Condition<L> {
    /// The constraint holding the degrees of freedom.
    fn held_constraint(&self) -> EqualityConstraint;
}

/// Boundary conditions on the degrees of freedom of a layout, whose constraints may depend on time.
///
/// The conditions are held in a nest of pairs, so that each is its own type. Starting from
/// [`BoundaryConditions`], each of the methods that hold degrees of freedom adds a condition.
/// The components a layout has are the only ones that can be held, and an axis that a layout
/// does not have is an error when compiling. The conditions borrow the sets they hold, such as
/// those of the model they are for, so they cannot outlive them. Each is eliminated from the
/// system unless it is made to be held by multipliers.
pub trait ConstrainAt<L> {
    /// The constraints holding the degrees of freedom at a time.
    fn constraint_at(&self, time: Quantity<Time>) -> EqualityConstraints;
    /// Gives `each` degree of freedom that is held at a time at a value other than the one that it
    /// starts at, with what that is, in the order of the constraints, so that the last that is
    /// given for a degree of freedom is the one that holds it.
    fn prescribed_at(&self, time: Quantity<Time>, each: &mut impl FnMut(usize, Prescription)) {
        self.constraint_at(time)
            .into_iter()
            .for_each(|constraint| each_prescribed(constraint, each))
    }
    /// Holds a component of every node in a set at its initial value.
    fn fix<'m, C, I>(self, component: C, nodes: &'m NodeSet) -> (Self, Fixed<'m, L>)
    where
        Self: Sized,
        L: Layout,
        C: Component<L, I>,
    {
        self.fix_by(Enforcement::Elimination, component, nodes)
    }
    /// Holds a component of every node in a set at its initial value, the way given.
    fn fix_by<'m, C, I>(
        self,
        enforcement: Enforcement,
        _: C,
        nodes: &'m NodeSet,
    ) -> (Self, Fixed<'m, L>)
    where
        Self: Sized,
        L: Layout,
        C: Component<L, I>,
    {
        (self, Fixed::new::<C, I>(Target::Set(nodes), enforcement))
    }
    /// Holds a component of the listed nodes at its initial value.
    fn fix_nodes<'m, C, I>(
        self,
        component: C,
        nodes: impl IntoIterator<Item = usize>,
    ) -> (Self, Fixed<'m, L>)
    where
        Self: Sized,
        L: Layout,
        C: Component<L, I>,
    {
        self.fix_nodes_by(Enforcement::Elimination, component, nodes)
    }
    /// Holds a component of the listed nodes at its initial value, the way given.
    fn fix_nodes_by<'m, C, I>(
        self,
        enforcement: Enforcement,
        _: C,
        nodes: impl IntoIterator<Item = usize>,
    ) -> (Self, Fixed<'m, L>)
    where
        Self: Sized,
        L: Layout,
        C: Component<L, I>,
    {
        let target = Target::Nodes(nodes.into_iter().collect());
        (self, Fixed::new::<C, I>(target, enforcement))
    }
    /// Holds a component of a node at its initial value.
    fn fix_node<'m, C, I>(self, component: C, node: usize) -> (Self, Fixed<'m, L>)
    where
        Self: Sized,
        L: Layout,
        C: Component<L, I>,
    {
        self.fix_node_by(Enforcement::Elimination, component, node)
    }
    /// Holds a component of a node at its initial value, the way given.
    fn fix_node_by<'m, C, I>(
        self,
        enforcement: Enforcement,
        component: C,
        node: usize,
    ) -> (Self, Fixed<'m, L>)
    where
        Self: Sized,
        L: Layout,
        C: Component<L, I>,
    {
        self.fix_nodes_by(enforcement, component, [node])
    }
    /// Holds a component of every node in a set at a value.
    fn prescribe<'m, C, I>(
        self,
        component: C,
        nodes: &'m NodeSet,
        value: C::Value,
    ) -> (Self, Prescribed<'m, L>)
    where
        Self: Sized,
        L: Layout,
        C: Component<L, I> + Prescribable,
    {
        self.prescribe_by(Enforcement::Elimination, component, nodes, value)
    }
    /// Holds a component of every node in a set at a value, the way given.
    fn prescribe_by<'m, C, I>(
        self,
        enforcement: Enforcement,
        _: C,
        nodes: &'m NodeSet,
        value: C::Value,
    ) -> (Self, Prescribed<'m, L>)
    where
        Self: Sized,
        L: Layout,
        C: Component<L, I> + Prescribable,
    {
        let prescribed =
            Prescribed::new::<C, I>(Target::Set(nodes), C::prescription(value), enforcement);
        (self, prescribed)
    }
    /// Holds a component of the listed nodes at a value.
    fn prescribe_nodes<'m, C, I>(
        self,
        component: C,
        nodes: impl IntoIterator<Item = usize>,
        value: C::Value,
    ) -> (Self, Prescribed<'m, L>)
    where
        Self: Sized,
        L: Layout,
        C: Component<L, I> + Prescribable,
    {
        self.prescribe_nodes_by(Enforcement::Elimination, component, nodes, value)
    }
    /// Holds a component of the listed nodes at a value, the way given.
    fn prescribe_nodes_by<'m, C, I>(
        self,
        enforcement: Enforcement,
        _: C,
        nodes: impl IntoIterator<Item = usize>,
        value: C::Value,
    ) -> (Self, Prescribed<'m, L>)
    where
        Self: Sized,
        L: Layout,
        C: Component<L, I> + Prescribable,
    {
        let target = Target::Nodes(nodes.into_iter().collect());
        let prescribed = Prescribed::new::<C, I>(target, C::prescription(value), enforcement);
        (self, prescribed)
    }
    /// Holds a component of a node at a value.
    fn prescribe_node<'m, C, I>(
        self,
        component: C,
        node: usize,
        value: C::Value,
    ) -> (Self, Prescribed<'m, L>)
    where
        Self: Sized,
        L: Layout,
        C: Component<L, I> + Prescribable,
    {
        self.prescribe_node_by(Enforcement::Elimination, component, node, value)
    }
    /// Holds a component of a node at a value, the way given.
    fn prescribe_node_by<'m, C, I>(
        self,
        enforcement: Enforcement,
        component: C,
        node: usize,
        value: C::Value,
    ) -> (Self, Prescribed<'m, L>)
    where
        Self: Sized,
        L: Layout,
        C: Component<L, I> + Prescribable,
    {
        self.prescribe_nodes_by(enforcement, component, [node], value)
    }
    /// Holds a component of every node in a set at a value that is a function of time.
    fn prescribe_with<'m, C, I>(
        self,
        component: C,
        nodes: &'m NodeSet,
        value: impl Fn(Quantity<Time>) -> C::Value,
    ) -> (Self, Timed<'m, L, impl Fn(Quantity<Time>) -> Prescription>)
    where
        Self: Sized,
        L: Layout,
        C: Component<L, I> + Prescribable,
    {
        self.prescribe_with_by(Enforcement::Elimination, component, nodes, value)
    }
    /// Holds a component of every node in a set at a value that is a function of time, the way
    /// given.
    fn prescribe_with_by<'m, C, I>(
        self,
        enforcement: Enforcement,
        component: C,
        nodes: &'m NodeSet,
        value: impl Fn(Quantity<Time>) -> C::Value,
    ) -> (Self, Timed<'m, L, impl Fn(Quantity<Time>) -> Prescription>)
    where
        Self: Sized,
        L: Layout,
        C: Component<L, I> + Prescribable,
    {
        (
            self,
            Timed::new::<C, I>(Target::Set(nodes), enforcement, component, value),
        )
    }
    /// Holds a component of the listed nodes at a value that is a function of time.
    fn prescribe_nodes_with<'m, C, I>(
        self,
        component: C,
        nodes: impl IntoIterator<Item = usize>,
        value: impl Fn(Quantity<Time>) -> C::Value,
    ) -> (Self, Timed<'m, L, impl Fn(Quantity<Time>) -> Prescription>)
    where
        Self: Sized,
        L: Layout,
        C: Component<L, I> + Prescribable,
    {
        self.prescribe_nodes_with_by(Enforcement::Elimination, component, nodes, value)
    }
    /// Holds a component of the listed nodes at a value that is a function of time, the way
    /// given.
    fn prescribe_nodes_with_by<'m, C, I>(
        self,
        enforcement: Enforcement,
        component: C,
        nodes: impl IntoIterator<Item = usize>,
        value: impl Fn(Quantity<Time>) -> C::Value,
    ) -> (Self, Timed<'m, L, impl Fn(Quantity<Time>) -> Prescription>)
    where
        Self: Sized,
        L: Layout,
        C: Component<L, I> + Prescribable,
    {
        let target = Target::Nodes(nodes.into_iter().collect());
        (
            self,
            Timed::new::<C, I>(target, enforcement, component, value),
        )
    }
}

/// Boundary conditions on the degrees of freedom of a layout, whose constraints do not depend on time.
///
/// A solver that is not a function of time takes these, so conditions that are a function of
/// time are not accepted by it.
pub trait Constrain<L> {
    /// The constraints holding the degrees of freedom.
    fn constraint(&self) -> EqualityConstraints;
}

/// No boundary conditions, to which others are added.
pub struct BoundaryConditions<L> {
    layout: PhantomData<L>,
}

impl<L> Default for BoundaryConditions<L> {
    fn default() -> Self {
        Self {
            layout: PhantomData,
        }
    }
}

impl<L> BoundaryConditions<L> {
    pub fn new() -> Self {
        Self::default()
    }
}

impl<L> ConstrainAt<L> for BoundaryConditions<L> {
    fn constraint_at(&self, _: Quantity<Time>) -> EqualityConstraints {
        Vec::new()
    }
    fn prescribed_at(&self, _: Quantity<Time>, _: &mut impl FnMut(usize, Prescription)) {}
}

impl<L> Constrain<L> for BoundaryConditions<L> {
    fn constraint(&self) -> EqualityConstraints {
        Vec::new()
    }
}

impl<L, T, C> ConstrainAt<L> for (T, C)
where
    T: ConstrainAt<L>,
    C: Condition<L>,
{
    fn constraint_at(&self, time: Quantity<Time>) -> EqualityConstraints {
        let mut constraints = self.0.constraint_at(time);
        constraints.push(self.1.constraint_at(time));
        constraints
    }
    fn prescribed_at(&self, time: Quantity<Time>, each: &mut impl FnMut(usize, Prescription)) {
        self.0.prescribed_at(time, each);
        self.1.prescribed_at(time, each);
    }
}

impl<L, T, C> Constrain<L> for (T, C)
where
    T: Constrain<L>,
    C: Static<L>,
{
    fn constraint(&self) -> EqualityConstraints {
        let mut constraints = self.0.constraint();
        constraints.push(self.1.held_constraint());
        constraints
    }
}
