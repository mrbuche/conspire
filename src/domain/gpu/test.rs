//! Tests that need a device. Where there is none they say so and pass, unless
//! `CONSPIRE_REQUIRE_GPU` is set, which makes a missing device or missing kernels a failure, so
//! that a run on a machine that has a device cannot pass by skipping.

use super::{Device, GpuError, kernels};

fn skip<T>(why: &str) -> Option<T> {
    assert!(
        std::env::var_os("CONSPIRE_REQUIRE_GPU").is_none(),
        "CONSPIRE_REQUIRE_GPU is set and {why}"
    );
    eprintln!("skipped: {why}");
    None
}

fn device() -> Option<Device> {
    match Device::new(0) {
        Ok(device) => Some(device),
        Err(GpuError::Unavailable(message)) => skip(&format!("there is no device: {message}")),
        Err(error) => panic!("{error}"),
    }
}

fn kernels_on(device: &Device, fused: bool) -> Option<super::Module> {
    match kernels::ptx(fused) {
        Ok(ptx) => Some(device.module_from_ptx(&ptx).unwrap()),
        Err(GpuError::KernelsNotBuilt) => skip("the kernels are not built"),
        Err(error) => panic!("{error}"),
    }
}

fn built_ptx() -> Option<alloc::borrow::Cow<'static, str>> {
    match kernels::ptx(false) {
        Ok(ptx) => Some(ptx),
        Err(GpuError::KernelsNotBuilt) => skip("the kernels are not built"),
        Err(error) => panic!("{error}"),
    }
}

#[test]
fn the_kernels_of_every_model_and_shape_are_in_the_ptx_under_the_name_that_the_host_looks_for() {
    use crate::constitutive::solid::{
        elastic::DeviceModel,
        hyperelastic::{NeoHookean, SaintVenantKirchhoff},
    };
    let Some(ptx) = built_ptx() else { return };
    for model in [NeoHookean::NAME, SaintVenantKirchhoff::NAME] {
        for (points, nodes) in [(1, 4), (4, 4), (8, 8), (6, 6), (8, 5)] {
            let name = kernels::block_forces(model, points, nodes);
            assert!(
                ptx.contains(&format!(".visible .entry {name}(")),
                "{name} is not in the PTX"
            );
        }
    }
    assert!(ptx.contains(".visible .entry squares("));
    assert!(ptx.contains(".visible .entry gather("));
}

#[test]
fn the_ptx_has_no_trap_but_that_of_a_panic_and_no_call_but_to_one() {
    let Some(ptx) = built_ptx() else { return };
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
fn the_device_is_opened_with_a_name_and_a_capability_that_the_kernels_are_built_for() {
    let Some(device) = device() else { return };
    eprintln!("{} {:?}", device.name(), device.capability());
    assert!(!device.name().is_empty());
    assert!(device.capability().0 >= 7);
}

#[test]
fn the_squares_kernel_gives_the_squares_with_and_without_fused_operations() {
    let Some(device) = device() else { return };
    for fused in [true, false] {
        let Some(module) = kernels_on(&device, fused) else {
            return;
        };
        let squares = module.function("squares").unwrap();
        let count = 5000_u32;
        let output = device.buffer::<f64>(count as usize).unwrap();
        device
            .launch(&squares, count as usize, 128, &[&output, &count])
            .unwrap();
        device.synchronize().unwrap();
        let mut values = vec![0.0; count as usize];
        output.download(&mut values).unwrap();
        values.iter().enumerate().for_each(|(index, &value)| {
            assert_eq!(value, ((index + 1) * (index + 1)) as f64, "{index}");
        });
    }
}

#[test]
fn a_kernel_that_is_not_there_is_an_error() {
    let Some(device) = device() else { return };
    let Some(module) = kernels_on(&device, true) else {
        return;
    };
    assert!(matches!(
        module.function("not_a_kernel"),
        Err(GpuError::Cuda(_))
    ));
}

#[test]
fn values_make_the_trip_to_the_device_and_back() {
    let Some(device) = device() else { return };
    let values = (0..1000)
        .map(|index| index as f64 * 0.5)
        .collect::<Vec<_>>();
    let buffer = device.buffer_of(&values).unwrap();
    let mut found = vec![0.0; values.len()];
    buffer.download(&mut found).unwrap();
    assert_eq!(found, values);
}

#[test]
fn a_buffer_and_a_kernel_outlive_their_device() {
    let Some(device) = device() else { return };
    let Some(module) = kernels_on(&device, true) else {
        return;
    };
    let squares = module.function("squares").unwrap();
    let output = device.buffer::<f64>(10).unwrap();
    drop(module);
    drop(device);
    let mut values = vec![-1.0; 10];
    output.download(&mut values).unwrap();
    drop(squares);
}

#[test]
fn a_device_is_used_from_many_threads() {
    let Some(device) = device() else { return };
    std::thread::scope(|scope| {
        (0..4).for_each(|thread| {
            let device = &device;
            scope.spawn(move || {
                let values = (0..100)
                    .map(|index| (thread * 100 + index) as f64)
                    .collect::<Vec<_>>();
                let buffer = device.buffer_of(&values).unwrap();
                let mut found = vec![0.0; values.len()];
                buffer.download(&mut found).unwrap();
                assert_eq!(found, values);
            });
        });
    });
}
