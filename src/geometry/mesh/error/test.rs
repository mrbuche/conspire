use super::MeshError;
use core::error::Error;

#[test]
fn the_message_of_a_pass_that_ran_out_says_how_far_it_got() {
    let message = MeshError::Budget {
        theta: 0.4,
        sweeps: 7,
    }
    .to_string();
    assert!(message.contains("all 7 sweeps"));
    assert!(message.contains("0.40"));
    let message = MeshError::Stall {
        theta: 0.4,
        sweeps: 3,
    }
    .to_string();
    assert!(message.contains("after 3 sweeps"));
    assert!(message.contains("0.40"));
}

#[test]
fn the_debug_of_an_error_is_its_message_on_a_line_of_its_own() {
    for error in [
        MeshError::Budget {
            theta: 0.4,
            sweeps: 7,
        },
        MeshError::HexahedraRequired,
        MeshError::Minimization,
        MeshError::Stall {
            theta: 0.4,
            sweeps: 3,
        },
        MeshError::SurfaceFit,
    ] {
        assert_eq!(format!("{error:?}"), format!("\n{error}\n"));
        assert!(!error.to_string().is_empty());
    }
}

#[test]
fn an_error_of_a_mesh_is_an_error() {
    fn is_an_error(_: &dyn Error) {}
    is_an_error(&MeshError::SurfaceFit);
}
