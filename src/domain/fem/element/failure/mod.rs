#[cfg(test)]
mod test;

use crate::fem::element::FiniteElementError;

/// The first failure of the elements of a block that a device found, in one word.
///
/// The threads of a device each compute an element, and record what failed with an atomic
/// minimum on a single word, so what they record is packed so that the smallest word is the
/// failure of the first element, and of the first integration point of it. The host then makes
/// the full error by computing that element again.
///
/// The bits, from the most significant: 32 of the element, 16 of the integration point, 8 of the
/// kind of the error (see [`ConstitutiveError::code`](crate::constitutive::ConstitutiveError::code)),
/// and 8 that are zero. A word of all ones is no failure.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct Failure(u64);

impl Failure {
    /// No failure, which is greater than every failure, so that a minimum keeps a failure.
    pub const NONE: Self = Self(u64::MAX);
    /// The failure of an integration point of an element, with the code of the error.
    pub const fn new(item: u32, point: u16, code: u8) -> Self {
        Self(((item as u64) << 32) | ((point as u64) << 16) | ((code as u64) << 8))
    }
    /// The failure of an element with an error.
    pub fn of(item: usize, error: &FiniteElementError) -> Self {
        Self::new(
            u32::try_from(item).unwrap_or(u32::MAX).min(u32::MAX - 1),
            u16::try_from(error.point.unwrap_or(0)).unwrap_or(u16::MAX),
            error.source.code(),
        )
    }
    /// The failure as the word that a device records.
    pub const fn bits(self) -> u64 {
        self.0
    }
    /// The failure from the word that a device recorded.
    pub const fn from_bits(bits: u64) -> Self {
        Self(bits)
    }
    /// Whether this is a failure, and not none.
    pub const fn is_failure(self) -> bool {
        self.0 != u64::MAX
    }
    /// The element that failed.
    pub const fn item(self) -> usize {
        (self.0 >> 32) as usize
    }
    /// The integration point of the element that failed.
    pub const fn point(self) -> usize {
        ((self.0 >> 16) & 0xFFFF) as usize
    }
    /// The code of the kind of error.
    pub const fn code(self) -> u8 {
        ((self.0 >> 8) & 0xFF) as u8
    }
}
