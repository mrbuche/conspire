#[cfg(test)]
mod test;

use crate::math::Scalar;
use std::thread::scope;

pub(super) const MINIMUM_PER_THREAD: usize = 1 << 14;

/// The values a source gives at `positions`, each added as many times as its count, into `values`.
///
/// The entries are computed in chunks, on up to `threads` threads including the calling one,
/// and never more threads than there are `minimum` entries for. Each entry is computed alone, so
/// the values are the same whatever the threads, and are what a host matrix sums for the pattern.
pub(super) fn fill(
    positions: &[(usize, usize)],
    counts: &[u32],
    source: &(dyn Fn(usize, usize) -> Scalar + Sync),
    values: &mut [Scalar],
    threads: usize,
    minimum: usize,
) {
    assert_eq!(positions.len(), values.len());
    assert_eq!(counts.len(), values.len());
    let threads = threads.min(values.len().div_ceil(minimum.max(1))).max(1);
    if threads == 1 {
        return chunk(positions, counts, source, values);
    }
    let length = values.len().div_ceil(threads);
    scope(|scope| {
        let mut chunks = values
            .chunks_mut(length)
            .zip(positions.chunks(length).zip(counts.chunks(length)));
        let first = chunks.next();
        chunks.for_each(|(values, (positions, counts))| {
            scope.spawn(move || chunk(positions, counts, source, values));
        });
        if let Some((values, (positions, counts))) = first {
            chunk(positions, counts, source, values)
        }
    });
}

fn chunk(
    positions: &[(usize, usize)],
    counts: &[u32],
    source: &(dyn Fn(usize, usize) -> Scalar + Sync),
    values: &mut [Scalar],
) {
    values
        .iter_mut()
        .zip(positions.iter().zip(counts))
        .for_each(|(value, (&(i, j), &count))| {
            let entry = source(i, j);
            *value = (0..count).fold(0.0, |sum, _| sum + entry)
        })
}
