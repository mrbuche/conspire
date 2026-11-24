#[cfg(test)]
mod test;

use crate::math::sparse::SparseError;

/// The matrix a device is given, in the order of its rows, and where its values come from.
///
/// Each entry of the device has one position that a source is asked for, and the number of times
/// that position is in the pattern, as a host matrix sums the duplicates of a pattern. A symmetric
/// matrix is given to the device by its upper triangle, which is the lower triangle that a source
/// is asked for, transposed.
#[derive(Debug, PartialEq)]
pub(super) struct Layout {
    /// The start of each row in `columns`, with one more at the end.
    pub(super) row_starts: Vec<i32>,
    /// The column of each entry of the device.
    pub(super) columns: Vec<i32>,
    /// The (row, column) a source is asked for the value of each entry of the device.
    pub(super) positions: Vec<(usize, usize)>,
    /// How many times each position is in the pattern.
    pub(super) counts: Vec<u32>,
}

/// An entry of the matrix of the device, with where it is and where its value comes from.
struct Entry {
    /// The (row, column) of the entry on the device.
    device: (usize, usize),
    /// The (row, column) a source is asked for its value.
    source: (usize, usize),
    /// How many times that position is in the pattern.
    count: u32,
}

impl Layout {
    /// The layout of `num` unknowns and the positions of `pattern`.
    ///
    /// Errors if `symmetric` is set and the pattern is not, or if the matrix has more rows or
    /// entries than 32-bit indices count.
    pub(super) fn new(
        num: usize,
        mut pattern: Vec<(usize, usize)>,
        symmetric: bool,
    ) -> Result<Self, SparseError> {
        assert!(
            !pattern.is_empty() || num == 0,
            "Matrix must have at least one entry."
        );
        assert!(
            pattern.iter().all(|&(i, j)| i < num && j < num),
            "Position out of bounds."
        );
        let too_large = || SparseError::Device("the matrix is too large for 32-bit indices".into());
        i32::try_from(num).map_err(|_| too_large())?;
        pattern.sort_unstable();
        let mut unique: Vec<((usize, usize), u32)> = Vec::with_capacity(pattern.len());
        for position in pattern {
            match unique.last_mut() {
                Some((last, count)) if *last == position => *count += 1,
                _ => unique.push((position, 1)),
            }
        }
        if symmetric
            && unique.iter().any(|&((i, j), _)| {
                unique
                    .binary_search_by_key(&(j, i), |entry| entry.0)
                    .is_err()
            })
        {
            return Err(SparseError::Unsymmetric);
        }
        let mut entries: Vec<Entry> = if symmetric {
            unique
                .into_iter()
                .filter(|&((i, j), _)| i >= j)
                .map(|((i, j), count)| Entry {
                    device: (j, i),
                    source: (i, j),
                    count,
                })
                .collect()
        } else {
            unique
                .into_iter()
                .map(|(position, count)| Entry {
                    device: position,
                    source: position,
                    count,
                })
                .collect()
        };
        if symmetric {
            entries.sort_unstable_by_key(|entry| entry.device);
        }
        i32::try_from(entries.len()).map_err(|_| too_large())?;
        let mut row_starts = vec![0_i32; num + 1];
        entries
            .iter()
            .for_each(|entry| row_starts[entry.device.0 + 1] += 1);
        (0..num).for_each(|row| row_starts[row + 1] += row_starts[row]);
        Ok(Self {
            row_starts,
            columns: entries.iter().map(|entry| entry.device.1 as i32).collect(),
            positions: entries.iter().map(|entry| entry.source).collect(),
            counts: entries.iter().map(|entry| entry.count).collect(),
        })
    }
}
