use super::{Application, GpuError, prepare, ptx};

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
    let ptx = prepare(Application::Elements, SAMPLE, true).unwrap();
    assert!(ptx.contains("mul.f64 \t%fd1, %fd2, %fd3;"));
    assert!(ptx.contains("add.f64 \t%fd4, %fd1, %fd5;"));
    assert!(ptx.contains("sub.f64 \t%fd6, %fd4, %fd7;"));
}

#[test]
fn fusing_keeps_the_rounding_of_everything_else() {
    let ptx = prepare(Application::Elements, SAMPLE, true).unwrap();
    for kept in ["fma.rn.f64", "div.rn.f64", "sqrt.rn.f64", "mul.rn.f32"] {
        assert!(ptx.contains(kept), "{kept}");
    }
}

#[test]
fn not_fusing_leaves_the_ptx_as_it_is() {
    assert_eq!(
        prepare(Application::Elements, SAMPLE, false).unwrap(),
        SAMPLE
    );
}

#[test]
fn kernels_that_are_not_built_are_an_error() {
    for source in ["", "  \n"] {
        for fused in [true, false] {
            assert!(matches!(
                prepare(Application::Elements, source, fused),
                Err(GpuError::KernelsNotBuilt(Application::Elements))
            ));
        }
    }
}

#[test]
fn the_kernels_that_are_built_are_embedded() {
    match ptx(Application::Elements, false) {
        Ok(ptx) => assert!(ptx.contains(".visible .entry squares"), "{ptx}"),
        Err(GpuError::KernelsNotBuilt(application)) => {
            eprintln!("skipped: the kernels of the {application} are not built");
        }
        Err(error) => panic!("{error}"),
    }
}

#[test]
fn the_error_of_kernels_that_are_not_built_names_them_and_how_to_build_them() {
    let message = GpuError::KernelsNotBuilt(Application::Elements).to_string();
    assert!(message.contains("elements"), "{message}");
    assert!(message.contains("kernels/build.sh elements"), "{message}");
}
