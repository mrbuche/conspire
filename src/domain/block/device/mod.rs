mod layout;

pub use layout::DeviceLayout;

/// A block of a model as the kernels take it: what kernel makes the forces of its elements, the
/// parameters of its model, what its elements share, and the layout of its elements.
pub struct DeviceBlock {
    /// The name of its model in the names of the kernels.
    pub model: &'static str,
    /// The parameters of the model.
    pub parameters: Vec<f64>,
    /// The integration points of each element.
    pub points: usize,
    /// The derivatives of the shape functions at the integration points of the parametric
    /// element, as point, then node, then component, which all of the elements share.
    pub derivatives: Vec<f64>,
    /// The weights of the integration points of the parametric element.
    pub weights: Vec<f64>,
    /// The layout of the elements.
    pub layout: DeviceLayout,
}

/// Something made of blocks that a device can run, in the order that the forces are added in.
pub trait DeviceBlocks {
    /// Adds the blocks that it is made of to `blocks`.
    fn device_blocks(&self, blocks: &mut Vec<DeviceBlock>);
}
