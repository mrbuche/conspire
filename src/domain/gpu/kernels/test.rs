use super::{GpuError, prepare};

const SAMPLE: &str = "\
mul.rn.f64 \t%fd1, %fd2, %fd3;
add.rn.f64 \t%fd4, %fd1, %fd5;
sub.rn.f64 \t%fd6, %fd4, %fd7;
fma.rn.f64 \t%fd8, %fd1, %fd2, %fd3;
div.rn.f64 \t%fd9, %fd1, %fd2;
sqrt.rn.f64 \t%fd10, %fd1;
mul.rn.f32 \t%f1, %f2, %f3;
";

#[test]
fn fusing_leaves_the_rounding_of_multiplications_additions_and_subtractions_unnamed() {
    let ptx = prepare(SAMPLE, true).unwrap();
    assert!(ptx.contains("mul.f64 \t%fd1, %fd2, %fd3;"));
    assert!(ptx.contains("add.f64 \t%fd4, %fd1, %fd5;"));
    assert!(ptx.contains("sub.f64 \t%fd6, %fd4, %fd7;"));
}

#[test]
fn fusing_keeps_the_rounding_of_everything_else() {
    let ptx = prepare(SAMPLE, true).unwrap();
    for kept in ["fma.rn.f64", "div.rn.f64", "sqrt.rn.f64", "mul.rn.f32"] {
        assert!(ptx.contains(kept), "{kept}");
    }
}

#[test]
fn not_fusing_leaves_the_ptx_as_it_is() {
    assert_eq!(prepare(SAMPLE, false).unwrap(), SAMPLE);
}

#[test]
fn kernels_that_are_not_built_are_an_error() {
    for source in ["", "  \n"] {
        for fused in [true, false] {
            assert!(matches!(
                prepare(source, fused),
                Err(GpuError::KernelsNotBuilt)
            ));
        }
    }
}
