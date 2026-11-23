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
        Ok(ptx) => {
            for name in ["squares", "gather"] {
                assert!(ptx.contains(&format!(".visible .entry {name}(")), "{name}");
            }
        }
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

#[cfg(feature = "elements")]
#[test]
fn the_kernels_of_every_model_and_shape_are_in_the_ptx_under_the_name_that_the_host_looks_for() {
    use super::block_forces;
    use crate::constitutive::{
        FromParameters,
        solid::hyperelastic::{NeoHookean, SaintVenantKirchhoff},
    };
    let Ok(ptx) = ptx(Application::Elements, false) else {
        return eprintln!("skipped: the kernels of the elements are not built");
    };
    for model in [NeoHookean::NAME, SaintVenantKirchhoff::NAME] {
        for (points, nodes) in [(1, 4), (4, 4), (8, 8), (6, 6), (8, 5)] {
            let name = block_forces(model, points, nodes);
            assert!(
                ptx.contains(&format!(".visible .entry {name}(")),
                "{name} is not in the PTX"
            );
        }
    }
}

#[cfg(feature = "elements")]
#[test]
fn the_kernels_of_every_model_for_elements_of_one_point_with_hourglass_control_are_in_the_ptx() {
    use super::block_forces_stabilized;
    use crate::constitutive::{
        FromParameters,
        solid::hyperelastic::{NeoHookean, SaintVenantKirchhoff},
    };
    let Ok(ptx) = ptx(Application::Elements, false) else {
        return eprintln!("skipped: the kernels of the elements are not built");
    };
    for model in [NeoHookean::NAME, SaintVenantKirchhoff::NAME] {
        let name = block_forces_stabilized(model, 8);
        assert!(
            ptx.contains(&format!(".visible .entry {name}(")),
            "{name} is not in the PTX"
        );
    }
}

#[test]
fn the_ptx_has_no_trap_but_that_of_a_panic_and_no_call_but_to_one() {
    let Ok(ptx) = ptx(Application::Elements, false) else {
        return eprintln!("skipped: the kernels of the elements are not built");
    };
    let traps = ptx.matches("trap;").count();
    assert!(
        traps <= 1,
        "{traps} traps: a function that a device does not have is reached"
    );
    assert!(
        !ptx.contains(".extern"),
        "the PTX needs what it does not have"
    );
    ptx.lines()
        .filter(|line| line.trim_start().starts_with("call"))
        .for_each(|line| assert!(line.contains("panicking"), "{line}"));
}

#[test]
fn the_second_pass_is_additions_only_with_no_call_no_trap_and_no_local_memory() {
    let Ok(ptx) = ptx(Application::Elements, false) else {
        return eprintln!("skipped: the kernels of the elements are not built");
    };
    let start = ptx
        .find(".visible .entry gather(")
        .expect("the kernel is there");
    let end = ptx[start..].find("\n}\n").expect("the kernel ends");
    let gather = &ptx[start..start + end];
    for absent in ["call", "trap;", ".local", "mul.rn.f64", "mul.f64", "fma."] {
        assert!(!gather.contains(absent), "{absent} is in the second pass");
    }
    assert!(gather.contains("add.rn.f64"));
}

const STEPS: [&str; 5] = [
    "kick_drift",
    "accelerate",
    "set_targets",
    "subtract_constant",
    "subtract_weighted",
];

fn kernel<'a>(ptx: &'a str, name: &str) -> &'a str {
    let start = ptx
        .find(&format!(".visible .entry {name}("))
        .expect("the kernel is there");
    let end = ptx[start..].find("\n}\n").expect("the kernel ends");
    &ptx[start..start + end]
}

#[test]
fn the_kernels_of_the_steps_are_embedded_under_their_names() {
    match ptx(Application::Dynamics, false) {
        Ok(ptx) => {
            for name in STEPS {
                assert!(ptx.contains(&format!(".visible .entry {name}(")), "{name}");
            }
        }
        Err(GpuError::KernelsNotBuilt(application)) => {
            eprintln!("skipped: the kernels of the {application} are not built");
        }
        Err(error) => panic!("{error}"),
    }
}

#[test]
fn the_kernels_of_the_steps_have_no_call_no_trap_and_no_local_memory() {
    let Ok(ptx) = ptx(Application::Dynamics, false) else {
        return eprintln!("skipped: the kernels of the steps are not built");
    };
    for name in STEPS {
        for absent in ["call", "trap;", ".local", "fma."] {
            assert!(!kernel(&ptx, name).contains(absent), "{absent} in {name}");
        }
    }
}

#[test]
fn the_first_half_of_a_step_is_the_operations_of_the_integrator_each_rounded() {
    let Ok(ptx) = ptx(Application::Dynamics, false) else {
        return eprintln!("skipped: the kernels of the steps are not built");
    };
    let kick_drift = kernel(&ptx, "kick_drift");
    assert_eq!(kick_drift.matches("mul.rn.f64").count(), 3);
    assert_eq!(kick_drift.matches("add.rn.f64").count(), 2);
    assert!(!kick_drift.contains("div."));
}

#[test]
fn the_accelerations_are_a_division_and_the_second_half_of_a_step() {
    let Ok(ptx) = ptx(Application::Dynamics, false) else {
        return eprintln!("skipped: the kernels of the steps are not built");
    };
    let accelerate = kernel(&ptx, "accelerate");
    assert_eq!(accelerate.matches("div.rn.f64").count(), 1);
    assert_eq!(accelerate.matches("mul.rn.f64").count(), 2);
    assert_eq!(accelerate.matches("add.rn.f64").count(), 1);
    assert!(accelerate.contains("neg.f64"));
}

#[test]
fn a_load_is_subtracted_as_the_host_subtracts_it_each_operation_rounded() {
    let Ok(ptx) = ptx(Application::Dynamics, false) else {
        return eprintln!("skipped: the kernels of the steps are not built");
    };
    let constant = kernel(&ptx, "subtract_constant");
    assert_eq!(constant.matches("sub.rn.f64").count(), 1);
    assert!(!constant.contains("mul.rn.f64"));
    let weighted = kernel(&ptx, "subtract_weighted");
    assert_eq!(weighted.matches("mul.rn.f64").count(), 3);
    assert_eq!(weighted.matches("sub.rn.f64").count(), 3);
}

#[test]
fn the_coordinates_that_are_moved_are_set_to_what_is_read_with_no_arithmetic() {
    let Ok(ptx) = ptx(Application::Dynamics, false) else {
        return eprintln!("skipped: the kernels of the steps are not built");
    };
    let set_targets = kernel(&ptx, "set_targets");
    for absent in ["mul.rn.f64", "add.rn.f64", "sub.rn.f64", "div.", "fma."] {
        assert!(!set_targets.contains(absent), "{absent}");
    }
}
