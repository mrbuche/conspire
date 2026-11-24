mod layout;

pub use layout::DeviceLayout;

/// What kind of element a block has, which is which kernel makes the forces of its elements.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DeviceKind {
    /// Elements that integrate at all of their points, with what the elements share.
    Full,
    /// Elements of one integration point, which have what they need stored for each of them.
    Stabilized,
}

/// A block of a model as the kernels take it: what kernel makes the forces of its elements, the
/// model parameters, what its elements share, what each has of its own, and the element layout.
pub struct DeviceBlock {
    /// What kind of element the block has.
    pub kind: DeviceKind,
    /// The name of its model in the names of the kernels.
    pub model: &'static str,
    /// The parameters of the model.
    pub parameters: Vec<f64>,
    /// The integration points of each element.
    pub points: usize,
    /// The derivatives of the shape functions at the integration points of the element.
    pub derivatives: Vec<f64>,
    /// The weights of the integration points of the parametric element.
    pub weights: Vec<f64>,
    /// The number of values that each element has of its own, which is none for the elements that
    /// share what they integrate with.
    pub values_per_item: usize,
    /// The values of the elements, as the value `k` of an element `item` at `k * items + item`,
    /// with the element last as in the layout.
    pub values: Vec<f64>,
    /// The layout of the elements.
    pub layout: DeviceLayout,
}

/// Something made of blocks that a device can run, in the order that the forces are added in.
pub trait DeviceBlocks {
    /// Adds the blocks that it is made of to `blocks`.
    fn device_blocks(&self, blocks: &mut Vec<DeviceBlock>);
}
