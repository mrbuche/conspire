use super::FloatMath;

fn close(a: f64, b: f64) {
    assert!((a - b).abs() <= 1e-14 * b.abs().max(1.0), "{a} is not {b}");
}

#[test]
fn the_cube_root_is_that_of_the_standard_library() {
    [27.0, 2.0, 1e-9, 1e12, 0.3, -8.0, 1.0]
        .iter()
        .for_each(|&x| {
            close(FloatMath::cbrt(x), f64::cbrt(x));
        });
    assert_eq!(FloatMath::cbrt(0.0), 0.0);
}

#[test]
fn an_integer_power_is_that_of_the_standard_library() {
    [(2.0, 10), (1.7, 3), (1.7, -3), (-1.3, 5), (5.0, 0)]
        .iter()
        .for_each(|&(x, n)| close(FloatMath::powi(x, n), f64::powi(x, n)));
}

#[test]
fn a_power_that_is_a_number_of_thirds_is_that_of_the_standard_library() {
    [
        (8.0, 2.0 / 3.0),
        (2.5, 5.0 / 3.0),
        (1.3, -2.0 / 3.0),
        (4.0, 1.0),
    ]
    .iter()
    .for_each(|&(x, p)| close(FloatMath::powf(x, p), f64::powf(x, p)));
}

#[test]
fn the_square_root_and_rounding_are_those_of_the_standard_library() {
    close(FloatMath::sqrt(2.0), f64::sqrt(2.0));
    assert_eq!(FloatMath::round(2.5), 3.0);
    assert_eq!(FloatMath::floor(-1.5), -2.0);
    assert_eq!(FloatMath::ceil(-1.5), -1.0);
}
