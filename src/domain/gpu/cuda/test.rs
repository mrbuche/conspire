use super::{GpuError, Library};

#[test]
fn a_library_that_is_not_there_is_unavailable() {
    assert!(matches!(
        Library::open(&["libconspire_not_there.so.0", "libconspire_neither.so"]),
        Err(GpuError::Unavailable(_))
    ));
}

#[test]
fn a_library_without_names_is_unavailable() {
    assert!(matches!(Library::open(&[]), Err(GpuError::Unavailable(_))));
}
