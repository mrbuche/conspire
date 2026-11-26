pub(crate) mod point_stiffness;

use crate::{domain::block::element::solid::SolidElement, math::Quantity, units::RateSquared};

/// Elements that bound their highest angular frequency, with the stiffness as `S` has it.
#[cfg_attr(not(feature = "elements"), allow(dead_code))]
pub trait CourantFriedrichsLewyElement<C, S>
where
    Self: SolidElement,
{
    type Masses;
    type Error;
    /// A bound on the highest angular frequency squared of the element with the lumped masses.
    fn squared_frequency_bound(
        &self,
        constitutive_model: &C,
        stiffness: &S,
        nodal_coordinates: &Self::Coordinates,
        nodal_masses: &Self::Masses,
    ) -> Result<Quantity<RateSquared>, Self::Error>;
}
