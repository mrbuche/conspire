use super::{
    Device, GpuError, Module,
    kernels::{self, Application},
};
use std::{env::var_os, thread::scope};

fn skip<T>(why: &str) -> Option<T> {
    assert!(
        var_os("CONSPIRE_REQUIRE_GPU").is_none(),
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

fn kernels_on(device: &Device, fused: bool) -> Option<Module> {
    match kernels::ptx(Application::Elements, fused) {
        Ok(ptx) => Some(device.module_from_ptx(&ptx).unwrap()),
        Err(GpuError::KernelsNotBuilt(application)) => {
            skip(&format!("the kernels of the {application} are not built"))
        }
        Err(error) => panic!("{error}"),
    }
}

#[test]
fn the_device_is_opened_with_a_name_and_a_capability() {
    let Some(device) = device() else { return };
    eprintln!("{} {:?}", device.name(), device.capability());
    assert!(!device.name().is_empty());
    assert!(device.capability().0 >= 7);
}

#[test]
fn values_make_the_trip_to_the_device_and_back() {
    let Some(device) = device() else { return };
    let values = (0..1000)
        .map(|index| index as f64 * 0.5)
        .collect::<Vec<_>>();
    let buffer = device.buffer_of(&values).unwrap();
    device.synchronize().unwrap();
    let mut found = vec![0.0; values.len()];
    buffer.download(&mut found).unwrap();
    assert_eq!(found, values);
}

#[test]
fn the_front_of_a_buffer_is_given_values_and_the_rest_is_left_as_it_was() {
    let Some(device) = device() else { return };
    let buffer = device.buffer_of(&[1.0, 2.0, 3.0, 4.0]).unwrap();
    buffer.upload_front(&[7.0, 8.0]).unwrap();
    buffer.upload_front(&[]).unwrap();
    let mut found = vec![0.0; 4];
    buffer.download(&mut found).unwrap();
    assert_eq!(found, [7.0, 8.0, 3.0, 4.0]);
}

#[test]
fn buffers_that_are_alive_have_addresses_that_are_their_own() {
    let Some(device) = device() else { return };
    let (first, second) = (
        device.buffer_of(&[1.0, 2.0]).unwrap(),
        device.buffer_of(&[3.0, 4.0]).unwrap(),
    );
    assert_ne!(first.address(), 0);
    assert_ne!(first.address(), second.address());
}

#[test]
fn a_buffer_outlives_its_device() {
    let Some(device) = device() else { return };
    let buffer = device.buffer_of(&[1.0, 2.0, 3.0]).unwrap();
    drop(device);
    let mut found = vec![0.0; 3];
    buffer.download(&mut found).unwrap();
    assert_eq!(found, [1.0, 2.0, 3.0]);
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
fn a_buffer_is_filled_with_words() {
    let Some(device) = device() else { return };
    let buffer = device.buffer::<u64>(5).unwrap();
    for word in [u32::MAX, 0, 0x01020304] {
        buffer.fill(word).unwrap();
        let mut found = vec![1; 5];
        buffer.download(&mut found).unwrap();
        let expected = (u64::from(word) << 32) | u64::from(word);
        assert_eq!(found, [expected; 5]);
    }
}

#[test]
fn the_errors_of_a_model_on_a_device_say_what_and_where() {
    let message = GpuError::NoKernel("block_forces_x_g1_n1".to_string()).to_string();
    assert!(message.contains("block_forces_x_g1_n1"), "{message}");
    let message = GpuError::Disagreement {
        block: 2,
        item: 7,
        point: 3,
    }
    .to_string();
    for part in ["integration point 3", "element 7", "block 2"] {
        assert!(message.contains(part), "{message}");
    }
}

#[test]
fn the_errors_of_a_run_of_dynamics_say_what_and_where() {
    let message = GpuError::Element {
        block: 1,
        item: 4,
        point: 3,
    }
    .to_string();
    for part in ["point 3", "element 4", "block 1", "run of dynamics"] {
        assert!(message.contains(part), "{message}");
    }
    let message = GpuError::Invalid("a run is started".to_string()).to_string();
    assert!(message.contains("a run is started"), "{message}");
}

#[test]
fn a_device_is_used_from_many_threads() {
    let Some(device) = device() else { return };
    scope(|scope| {
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
