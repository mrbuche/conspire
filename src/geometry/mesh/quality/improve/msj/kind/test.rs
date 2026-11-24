use super::kind_of;
use crate::math::{
    Scalar,
    assert::{Assert, AssertionError},
};

#[test]
fn each_kind_counts_its_points_and_edges() {
    assert_eq!(
        (0..4)
            .map(kind_of)
            .map(|kind| (kind.points.len(), kind.edges.len()))
            .collect::<Vec<_>>(),
        [(9, 12), (4, 6), (6, 9), (8, 8)]
    )
}

#[test]
fn the_points_of_every_kind_are_differences_of_nodes() -> Result<(), AssertionError> {
    for kind in (0..4).map(kind_of) {
        for point in kind.points {
            for row in &point.weights {
                Assert::default().eq_within_tols(row.iter().sum::<Scalar>(), 0.0)?;
            }
        }
    }
    Ok(())
}
