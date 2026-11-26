#[cfg(test)]
mod test;

/// The first step of a run that a device found to be unstable, and the first element of a block
/// that was, in one word.
///
/// The threads of a device each check an element, and record one that is unstable with an atomic
/// minimum on a single word, so the step is in the most significant half, and the element in the
/// other. A word of all ones is none.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct UnstableRecord(u64);

impl UnstableRecord {
    /// No unstable element, which is greater than every record, so that a minimum keeps one.
    pub const NONE: Self = Self(u64::MAX);
    /// The record of an element that is unstable at a step.
    pub fn of(step: u32, item: usize) -> Self {
        let item = u32::try_from(item).unwrap_or(u32::MAX).min(u32::MAX - 1);
        Self(((step as u64) << 32) | item as u64)
    }
    /// The record as the word that a device records.
    pub const fn bits(self) -> u64 {
        self.0
    }
    /// The record from the word that a device recorded.
    pub const fn from_bits(bits: u64) -> Self {
        Self(bits)
    }
    /// Whether this is the record of an unstable element, and not none.
    pub const fn is_unstable(self) -> bool {
        self.0 != u64::MAX
    }
    /// The step at which the element is unstable.
    pub const fn step(self) -> u32 {
        (self.0 >> 32) as u32
    }
    /// The element that is unstable.
    pub const fn item(self) -> usize {
        (self.0 & 0xFFFF_FFFF) as usize
    }
}
