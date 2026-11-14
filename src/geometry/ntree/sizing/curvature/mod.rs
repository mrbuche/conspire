use crate::{
    math::{Quantity, Scalar},
    units::Length,
};

/// Parameters controlling curvature-driven octree refinement.
pub struct CurvatureSizing {
    /// Absolute Dunyach chord-error tolerance.
    pub tolerance: Option<Quantity<Length>>,
    /// Lipschitz grading rate for sizing fields.
    pub gradation: Scalar,
    /// Minimum cell size relative to bounding box.
    pub floor_fraction: Scalar,
    /// Absolute lower bound on the local thickness that drives refinement.
    pub thickness_floor: Option<Quantity<Length>>,
}

impl Default for CurvatureSizing {
    fn default() -> Self {
        Self {
            tolerance: None,
            gradation: 0.5,
            floor_fraction: 1.0e-3,
            thickness_floor: None,
        }
    }
}
