use super::{Device, DeviceSparseSolver};
use crate::math::{
    Scalar, Tensor, Vector,
    assert::{Assert, AssertionError},
    sparse::{SparseError, SparseSolver},
};
use std::env::var_os;

fn skip<T>(why: &str) -> Option<T> {
    assert!(
        var_os("CONSPIRE_REQUIRE_GPU").is_none(),
        "CONSPIRE_REQUIRE_GPU is set and {why}"
    );
    eprintln!("skipped: {why}");
    None
}

fn solver(num: usize, pattern: &[(usize, usize)], symmetric: bool) -> Option<DeviceSparseSolver> {
    let device = match Device::new(0) {
        Ok(device) => device,
        Err(error) => return skip(&format!("there is no device: {error}")),
    };
    match DeviceSparseSolver::from_pattern(&device, num, pattern.to_vec(), symmetric) {
        Ok(solver) => Some(solver),
        Err(SparseError::Device(message)) if message.contains("is available") => skip(&message),
        Err(error) => panic!("{error}"),
    }
}

fn product(
    pattern: &[(usize, usize)],
    source: impl Fn(usize, usize) -> Scalar,
    x: &Vector,
) -> Vector {
    let mut product = Vector::zero(x.len());
    pattern
        .iter()
        .for_each(|&(i, j)| product[i] += source(i, j) * x[j]);
    product
}

fn loose() -> Assert {
    Assert {
        abs_tol: 1e-9,
        rel_tol: 1e-9,
        ..Assert::default()
    }
}

fn right_hand_side(num: usize) -> Vector {
    (0..num).map(|i| 1.0 + 0.1 * (i % 7) as Scalar).collect()
}

mod asymmetric {
    use super::*;

    const N: usize = 100;

    fn pattern() -> Vec<(usize, usize)> {
        let mut pattern: Vec<(usize, usize)> = (0..N).map(|i| (i, i)).collect();
        (0..N).for_each(|i| {
            pattern.push((i, (3 * i + 1) % N));
            pattern.push(((5 * i + 2) % N, i));
        });
        pattern
    }

    fn values(scale: Scalar) -> impl Fn(usize, usize) -> Scalar {
        move |i, j| {
            if i == j {
                8.0
            } else {
                scale * (((i * 7 + j * 3) % 5) as Scalar / 5.0 - 0.4)
            }
        }
    }

    #[test]
    fn solves_and_refactors_like_the_host() -> Result<(), AssertionError> {
        let pattern = pattern();
        let Some(mut device) = solver(N, &pattern, false) else {
            return Ok(());
        };
        let host = SparseSolver::from_pattern(N, pattern.clone(), false);
        let b = right_hand_side(N);
        for scale in [1.0, 2.5, 0.3] {
            let x = device.solve(values(scale), &b)?;
            Assert::default().eq_within_tols(product(&pattern, values(scale), &x), &b)?;
            Assert::default().eq_within_tols(&x, &host.solve(values(scale), &b)?)?;
        }
        Ok(())
    }

    #[test]
    fn an_asymmetric_pattern_is_not_symmetric() {
        let pattern = pattern();
        let Some(device) = Device::new(0).ok() else {
            return;
        };
        assert_eq!(
            DeviceSparseSolver::from_pattern(&device, N, pattern, true).err(),
            Some(SparseError::Unsymmetric)
        );
    }
}

mod symmetric {
    use super::*;

    const SIDE: usize = 12;
    const N: usize = SIDE * SIDE;

    fn pattern() -> Vec<(usize, usize)> {
        let mut pattern = Vec::new();
        (0..SIDE).for_each(|x| {
            (0..SIDE).for_each(|y| {
                let i = SIDE * x + y;
                pattern.push((i, i));
                if x + 1 < SIDE {
                    pattern.push((i, i + SIDE));
                    pattern.push((i + SIDE, i));
                }
                if y + 1 < SIDE {
                    pattern.push((i, i + 1));
                    pattern.push((i + 1, i));
                }
            })
        });
        pattern
    }

    fn laplacian(shift: Scalar) -> impl Fn(usize, usize) -> Scalar {
        move |i, j| if i == j { 4.0 - shift } else { -1.0 }
    }

    #[test]
    fn solves_a_positive_definite_system() -> Result<(), AssertionError> {
        let pattern = pattern();
        let Some(mut device) = solver(N, &pattern, true) else {
            return Ok(());
        };
        let b = right_hand_side(N);
        let x = device.solve(laplacian(0.0), &b)?;
        Assert::default().eq_within_tols(product(&pattern, laplacian(0.0), &x), &b)?;
        let host = SparseSolver::from_pattern(N, pattern, true);
        Assert::default().eq_within_tols(&x, &host.solve(laplacian(0.0), &b)?)
    }

    #[test]
    fn solves_an_indefinite_system_and_refactors() -> Result<(), AssertionError> {
        let pattern = pattern();
        let Some(mut device) = solver(N, &pattern, true) else {
            return Ok(());
        };
        let host = SparseSolver::from_pattern(N, pattern.clone(), true);
        let b = right_hand_side(N);
        for shift in [2.3, 0.5, 3.1] {
            let x = device.solve(laplacian(shift), &b)?;
            loose().eq_within_tols(product(&pattern, laplacian(shift), &x), &b)?;
            loose().eq_within_tols(&x, &host.solve(laplacian(shift), &b)?)?;
        }
        Ok(())
    }

    #[test]
    fn entries_of_a_duplicated_position_are_summed() -> Result<(), AssertionError> {
        let mut pattern = pattern();
        pattern.push((0, 0));
        let Some(mut device) = solver(N, &pattern, true) else {
            return Ok(());
        };
        let b = right_hand_side(N);
        // The source is asked for each entry of the pattern, so the diagonal at zero counts twice.
        let x = device.solve(laplacian(0.0), &b)?;
        Assert::default().eq_within_tols(product(&pattern, laplacian(0.0), &x), &b)
    }
}
