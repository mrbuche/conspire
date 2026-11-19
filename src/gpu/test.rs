use super::{Device, GpuError};
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
fn a_buffer_outlives_its_device() {
    let Some(device) = device() else { return };
    let buffer = device.buffer_of(&[1.0, 2.0, 3.0]).unwrap();
    drop(device);
    let mut found = vec![0.0; 3];
    buffer.download(&mut found).unwrap();
    assert_eq!(found, [1.0, 2.0, 3.0]);
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
