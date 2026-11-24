use super::Layout;
use crate::math::sparse::SparseError;

#[test]
fn a_general_matrix_has_the_rows_of_the_sorted_pattern() {
    let pattern = vec![(2, 2), (0, 1), (1, 1), (2, 0), (0, 0), (1, 2)];
    let layout = Layout::new(3, pattern, false).unwrap();
    assert_eq!(layout.row_starts, [0, 2, 4, 6]);
    assert_eq!(layout.columns, [0, 1, 1, 2, 0, 2]);
    assert_eq!(
        layout.positions,
        [(0, 0), (0, 1), (1, 1), (1, 2), (2, 0), (2, 2)]
    );
    assert_eq!(layout.counts, [1; 6]);
}

#[test]
fn duplicates_are_counted_once_each_at_their_position() {
    let pattern = vec![(0, 0), (1, 1), (0, 0), (1, 0), (0, 0), (1, 1)];
    let layout = Layout::new(2, pattern, false).unwrap();
    assert_eq!(layout.positions, [(0, 0), (1, 0), (1, 1)]);
    assert_eq!(layout.counts, [3, 1, 2]);
    assert_eq!(layout.row_starts, [0, 1, 3]);
}

#[test]
fn a_symmetric_matrix_has_its_upper_triangle_from_the_lower_one() {
    let pattern = vec![(0, 0), (0, 1), (1, 0), (1, 1), (1, 2), (2, 1), (2, 2)];
    let layout = Layout::new(3, pattern, true).unwrap();
    assert_eq!(layout.row_starts, [0, 2, 4, 5]);
    assert_eq!(layout.columns, [0, 1, 1, 2, 2]);
    assert_eq!(layout.positions, [(0, 0), (1, 0), (1, 1), (2, 1), (2, 2)]);
    assert_eq!(layout.counts, [1; 5]);
}

#[test]
fn a_symmetric_matrix_counts_the_duplicates_of_the_lower_triangle_alone() {
    let pattern = vec![(0, 0), (1, 0), (1, 0), (0, 1), (0, 1), (0, 1), (1, 1)];
    let layout = Layout::new(2, pattern, true).unwrap();
    assert_eq!(layout.positions, [(0, 0), (1, 0), (1, 1)]);
    assert_eq!(layout.counts, [1, 2, 1]);
}

#[test]
fn a_pattern_that_is_not_symmetric_is_refused_for_a_symmetric_matrix() {
    let pattern = vec![(0, 0), (0, 1), (1, 1)];
    assert_eq!(Layout::new(2, pattern, true), Err(SparseError::Unsymmetric));
}

#[test]
fn no_unknowns_is_no_entries() {
    let layout = Layout::new(0, Vec::new(), false).unwrap();
    assert_eq!(layout.row_starts, [0]);
    assert!(layout.columns.is_empty());
}

#[test]
#[should_panic(expected = "Position out of bounds.")]
fn a_position_out_of_bounds_is_refused() {
    let _ = Layout::new(2, vec![(0, 0), (2, 0)], false);
}
