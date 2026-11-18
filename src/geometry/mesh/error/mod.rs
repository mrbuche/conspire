#[cfg(test)]
mod test;

use crate::math::Scalar;
use core::{
    error::Error,
    fmt::{self, Debug, Display, Formatter},
};

/// Possible errors for meshes.
pub enum MeshError {
    Budget { theta: Scalar, sweeps: usize },
    HexahedraRequired,
    Minimization,
    Stall { theta: Scalar, sweeps: usize },
    SurfaceFit,
}

impl Display for MeshError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Budget { theta, sweeps } => write!(
                f,
                "The scaled Jacobian pass used all {sweeps} sweeps without reaching {theta:.2}."
            ),
            Self::HexahedraRequired => write!(
                f,
                "The mesh must consist of hexahedra, tetrahedra, pyramids or wedges."
            ),
            Self::Minimization => write!(f, "Could not minimize the energy."),
            Self::Stall { theta, sweeps } => write!(
                f,
                "The scaled Jacobian pass stalled after {sweeps} sweeps short of {theta:.2}."
            ),
            Self::SurfaceFit => write!(f, "Could not fit the surface."),
        }
    }
}

impl Debug for MeshError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "\n{self}\n")
    }
}

impl Error for MeshError {}
