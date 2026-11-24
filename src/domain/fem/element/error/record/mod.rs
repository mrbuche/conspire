#[cfg(test)]
mod test;

use super::FiniteElementError;

/// The first error of the elements of a block that a device found, in one word.
///
/// The threads of a device each compute an element, and record what went wrong with an atomic
/// minimum on a single word, so what they record is packed so that the smallest word is the
/// error of the first element, and of the first integration point of it. The host then makes
/// the full error by computing that element again.
///
/// The bits, from the most significant: 32 of the element, 16 of the integration point, 8 of the
/// kind of the error (see [`ConstitutiveError::code`](crate::constitutive::ConstitutiveError::code)),
/// and 8 that are zero. A word of all ones is no error.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct ErrorRecord(u64);

impl ErrorRecord {
    /// No error, which is greater than every record, so that a minimum keeps an error.
    pub const NONE: Self = Self(u64::MAX);
    /// The record of an error at an integration point of an element, with the code of the error.
    pub const fn new(item: u32, point: u16, code: u8) -> Self {
        Self(((item as u64) << 32) | ((point as u64) << 16) | ((code as u64) << 8))
    }
    /// The record of the error of an element.
    pub fn of(item: usize, error: &FiniteElementError) -> Self {
        Self::new(
            u32::try_from(item).unwrap_or(u32::MAX).min(u32::MAX - 1),
            u16::try_from(error.point.unwrap_or(0)).unwrap_or(u16::MAX),
            error.source.code(),
        )
    }
    /// The record as the word that a device records.
    pub const fn bits(self) -> u64 {
        self.0
    }
    /// The record from the word that a device recorded.
    pub const fn from_bits(bits: u64) -> Self {
        Self(bits)
    }
    /// Whether this is the record of an error, and not none.
    pub const fn is_error(self) -> bool {
        self.0 != u64::MAX
    }
    /// The element that has the error.
    pub const fn item(self) -> usize {
        (self.0 >> 32) as usize
    }
    /// The integration point of the element that has the error.
    pub const fn point(self) -> usize {
        ((self.0 >> 16) & 0xFFFF) as usize
    }
    /// The code of the kind of error.
    pub const fn code(self) -> u8 {
        ((self.0 >> 8) & 0xFF) as u8
    }
}
