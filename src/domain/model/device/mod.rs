#[cfg(feature = "fem")]
mod resident;

#[cfg(feature = "fem")]
pub use resident::{DeviceError, DeviceStability, ExplicitRun, Resident};

use super::{Blocks, Model};
use crate::domain::block::device::{DeviceBlock, DeviceBlocks};

impl<B1, B2> DeviceBlocks for Blocks<B1, B2>
where
    B1: DeviceBlocks,
    B2: DeviceBlocks,
{
    fn device_blocks(&self, blocks: &mut Vec<DeviceBlock>) {
        self.0.device_blocks(blocks);
        self.1.device_blocks(blocks)
    }
}

impl<B> DeviceBlocks for Model<B, 3>
where
    B: DeviceBlocks,
{
    fn device_blocks(&self, blocks: &mut Vec<DeviceBlock>) {
        self.blocks().device_blocks(blocks)
    }
}
