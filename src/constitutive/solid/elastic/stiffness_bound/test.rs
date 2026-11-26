use super::StiffnessBound;
use crate::{
    constitutive::{
        ConstitutiveError,
        solid::hyperelastic::{NeoHookean, SaintVenantKirchhoff},
    },
    math::{
        Scalar, SquareMatrix,
        assert::{Assert, AssertionError},
    },
    mechanics::DeformationGradient,
    units::Stress,
};
use core::array::from_fn;

const BULK_MODULI: [Scalar; 4] = [0.1, 0.7, 5.0 / 3.0, 100.0];
const DEFORMATIONS: u64 = 500;

fn random(index: u64) -> Scalar {
    ((index.wrapping_mul(2654435761).wrapping_add(index >> 3)) % 4294967296) as Scalar
        / 4294967296.0
        - 0.5
}

fn deformations() -> impl Iterator<Item = DeformationGradient> {
    (0..DEFORMATIONS).filter_map(|trial| {
        let amount = 0.05 + 1.5 * (trial as Scalar / DEFORMATIONS as Scalar);
        let deformation_gradient: DeformationGradient = from_fn(|i| {
            from_fn(|j| {
                (i == j) as u8 as Scalar + amount * random(1 + 9 * trial + 3 * i as u64 + j as u64)
            })
        })
        .into();
        (deformation_gradient.determinant() > 0.15).then_some(deformation_gradient)
    })
}

fn bounds<C>(
    model: &C,
    deformation_gradient: &DeformationGradient,
) -> Result<bool, ConstitutiveError>
where
    C: StiffnessBound,
{
    let (first, second) = model.stiffness_bound(deformation_gradient)?;
    let stiffness = model.first_piola_kirchhoff_tangent_stiffness(deformation_gradient)?;
    let jacobian = deformation_gradient.determinant();
    let inverse = deformation_gradient.inverse_transpose();
    let tolerance = 1e-9 * (model.bulk_modulus().value() + model.shear_modulus().value());
    let mut difference = SquareMatrix::zero(9);
    for row in 0..9 {
        let (i, j) = (row / 3, row % 3);
        for column in 0..9 {
            let (k, l) = (column / 3, column % 3);
            let mut entry = first.value() * inverse[i][j].value() * inverse[k][l].value()
                - stiffness[i][j][k][l].value() / jacobian;
            if i == k {
                entry += 2.0
                    * second.value()
                    * (0..3)
                        .map(|m| inverse[m][j].value() * inverse[m][l].value())
                        .sum::<Scalar>();
            }
            difference[row][column] = jacobian * entry;
        }
        difference[row][row] += tolerance;
    }
    Ok(difference
        .factorize_ldl()
        .is_ok_and(|decomposition| decomposition.inertia() == (9, 0, 0)))
}

macro_rules! test_stiffness_bound {
    ($module:ident, $model:ident) => {
        mod $module {
            use super::*;

            fn model(bulk_modulus: Scalar) -> $model {
                $model {
                    bulk_modulus: Stress::pascals(bulk_modulus),
                    shear_modulus: Stress::pascals(1.0),
                }
            }

            #[test]
            fn is_the_lame_moduli_in_the_reference_configuration() -> Result<(), AssertionError> {
                let identity: DeformationGradient =
                    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]].into();
                for bulk_modulus in BULK_MODULI {
                    let (first, second) = model(bulk_modulus).stiffness_bound(&identity)?;
                    Assert::default().eq_within_tols(
                        &first,
                        &Stress::pascals((bulk_modulus - 2.0 / 3.0).max(0.0)),
                    )?;
                    Assert::default().eq_within_tols(&second, &Stress::pascals(1.0))?;
                }
                Ok(())
            }

            #[test]
            fn bounds_the_tangent_stiffness() -> Result<(), AssertionError> {
                for bulk_modulus in BULK_MODULI {
                    let model = model(bulk_modulus);
                    for deformation_gradient in deformations() {
                        assert!(
                            bounds(&model, &deformation_gradient)?,
                            "bulk modulus {bulk_modulus}, {deformation_gradient:?}"
                        );
                    }
                }
                Ok(())
            }
        }
    };
}

test_stiffness_bound!(neo_hookean, NeoHookean);
test_stiffness_bound!(saint_venant_kirchhoff, SaintVenantKirchhoff);
