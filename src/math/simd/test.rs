use super::{
    axpy, axpy_scalar, rank_one_quad, rank_one_quad_scalar, rank_two_quad, rank_two_quad_scalar,
};
use crate::math::Scalar;

const BACKING: usize = 16;
const LENGTHS: [usize; 7] = [0, 3, 4, 5, 8, 9, 12];

fn backing(length: usize, seed: usize) -> Vec<Scalar> {
    (0..BACKING)
        .map(|index| {
            if index < length {
                ((index + seed) % 5 + 1) as Scalar
            } else {
                0.5
            }
        })
        .collect()
}

#[test]
fn axpy_is_that_of_the_shorter_slice() {
    for &target_length in &LENGTHS {
        for &column_length in &LENGTHS {
            let column = backing(column_length, 1);
            let mut found = backing(target_length, 2);
            let mut expected = found.clone();
            axpy(&mut found[..target_length], &column[..column_length], 3.0);
            axpy_scalar(
                &mut expected[..target_length],
                &column[..column_length],
                3.0,
            );
            assert_eq!(found, expected, "{target_length} {column_length}");
        }
    }
}

#[test]
fn a_rank_one_update_is_that_of_the_shortest_slice() {
    for index in 0..LENGTHS.len() {
        let length = |offset: usize| LENGTHS[(index + offset) % LENGTHS.len()];
        let column = backing(length(4), 1);
        let mut found = [0, 1, 2, 3].map(|offset| backing(length(offset), offset));
        let mut expected = found.clone();
        let [a, b, c, d] = &mut found;
        rank_one_quad(
            &mut a[..length(0)],
            &mut b[..length(1)],
            &mut c[..length(2)],
            &mut d[..length(3)],
            &column[..length(4)],
            [1.0, 2.0, 3.0, 4.0],
        );
        let [a, b, c, d] = &mut expected;
        rank_one_quad_scalar(
            &mut a[..length(0)],
            &mut b[..length(1)],
            &mut c[..length(2)],
            &mut d[..length(3)],
            &column[..length(4)],
            [1.0, 2.0, 3.0, 4.0],
        );
        assert_eq!(found, expected, "{index}");
    }
}

#[test]
fn a_rank_two_update_is_that_of_the_shortest_slice() {
    for index in 0..LENGTHS.len() {
        let length = |offset: usize| LENGTHS[(index + offset) % LENGTHS.len()];
        let column = backing(length(4), 1);
        let other = backing(length(5), 2);
        let mut found = [0, 1, 2, 3].map(|offset| backing(length(offset), offset));
        let mut expected = found.clone();
        let [a, b, c, d] = &mut found;
        rank_two_quad(
            &mut a[..length(0)],
            &mut b[..length(1)],
            &mut c[..length(2)],
            &mut d[..length(3)],
            &column[..length(4)],
            &other[..length(5)],
            [1.0, 2.0, 3.0, 4.0],
            [4.0, 3.0, 2.0, 1.0],
        );
        let [a, b, c, d] = &mut expected;
        rank_two_quad_scalar(
            &mut a[..length(0)],
            &mut b[..length(1)],
            &mut c[..length(2)],
            &mut d[..length(3)],
            &column[..length(4)],
            &other[..length(5)],
            [1.0, 2.0, 3.0, 4.0],
            [4.0, 3.0, 2.0, 1.0],
        );
        assert_eq!(found, expected, "{index}");
    }
}
