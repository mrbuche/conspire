use super::squared_frequency_bound_from;
use crate::{
    constitutive::solid::{
        elastic::Elastic,
        hyperelastic::{NeoHookean, SaintVenantKirchhoff},
    },
    domain::block::element::time_scale::courant_friedrichs_lewy::point_stiffness::ModelBound,
    fem::element::{
        ElementNodalCoordinates, ElementNodalReferenceCoordinates, FiniteElement,
        FiniteElementError,
        linear::{Hexahedron, Pyramid, Tetrahedron, Wedge},
        mass::{ElementNodalLumpedMasses, LumpedMassFiniteElement},
        solid::elastic::ElasticFiniteElement,
        stabilized::{self, StabilizedElement},
        time_scale::{CourantFriedrichsLewyElement, largest_eigenvalue},
    },
    math::Scalar,
    units::{Density, Stress},
};
use core::array::from_fn;

const TRIALS: u64 = 100;

fn random(index: u64) -> Scalar {
    ((index.wrapping_mul(2654435761).wrapping_add(index >> 3)) % 4294967296) as Scalar
        / 4294967296.0
        - 0.5
}

fn neo_hookean(bulk_modulus: Scalar) -> NeoHookean {
    NeoHookean {
        shear_modulus: Stress::pascals(1.0),
        bulk_modulus: Stress::pascals(bulk_modulus),
    }
}

fn saint_venant_kirchhoff(bulk_modulus: Scalar) -> SaintVenantKirchhoff {
    SaintVenantKirchhoff {
        shear_modulus: Stress::pascals(1.0),
        bulk_modulus: Stress::pascals(bulk_modulus),
    }
}

fn nodes<const N: usize>(
    parametric: [[Scalar; 3]; N],
    trial: u64,
    distortion: Scalar,
    strain: Scalar,
) -> ([[Scalar; 3]; N], [[Scalar; 3]; N]) {
    let seed = 1000 * trial;
    let reference: [[Scalar; 3]; N] = from_fn(|a| {
        from_fn(|i| parametric[a][i] + distortion * random(seed + (3 * a + i) as u64 + 1))
    });
    let current = reference.map(|node| {
        from_fn(|i| {
            node[i]
                + (0..3)
                    .map(|j| strain * random(seed + 500 + (3 * i + j) as u64) * node[j])
                    .sum::<Scalar>()
        })
    });
    (reference, current)
}

fn ratios<F, C, const G: usize, const N: usize>(
    constitutive_model: &C,
    distortion: Scalar,
    strain: Scalar,
) -> Vec<Scalar>
where
    C: Elastic,
    F: From<ElementNodalReferenceCoordinates<N>>
        + FiniteElement<G, 3, N, N>
        + LumpedMassFiniteElement<G, 3, N, N>
        + ElasticFiniteElement<C, G, 3, N, N>
        + CourantFriedrichsLewyElement<
            C,
            ModelBound,
            Masses = ElementNodalLumpedMasses<N>,
            Error = FiniteElementError,
        >,
{
    let parametric: [[Scalar; 3]; N] =
        from_fn(|a| from_fn(|i| F::parametric_reference()[a][i].value()));
    let mut ratios: Vec<Scalar> = (0..TRIALS)
        .map(|trial| {
            let (reference, current) = nodes(parametric, trial, distortion, strain);
            let element = F::from(ElementNodalReferenceCoordinates::from(reference));
            let coordinates = ElementNodalCoordinates::from(current);
            let masses =
                element.nodal_lumped_masses(&[Density::kilograms_per_cubic_meter(1.0); G].into());
            let exact = largest_eigenvalue(
                &element
                    .nodal_stiffnesses(constitutive_model, &coordinates)
                    .unwrap(),
                &masses,
            );
            let bound = element
                .squared_frequency_bound(constitutive_model, &ModelBound, &coordinates, &masses)
                .unwrap()
                .value();
            assert!(
                bound >= exact * (1.0 - 1e-9),
                "trial {trial}: the bound {bound} is below {exact}"
            );
            (bound / exact).sqrt()
        })
        .collect();
    ratios.sort_by(|a, b| a.partial_cmp(b).unwrap());
    ratios
}

macro_rules! test_bounds {
    ($name:ident, $element:ty, $points:literal, $nodes:literal, $loosest:literal) => {
        #[test]
        fn $name() {
            for bulk_modulus in [5.0 / 3.0, 100.0] {
                for (distortion, strain) in [(0.0, 0.0), (0.2, 0.0), (0.2, 0.1), (0.2, 0.4)] {
                    let neo = ratios::<$element, _, $points, $nodes>(
                        &neo_hookean(bulk_modulus),
                        distortion,
                        strain,
                    );
                    let svk = ratios::<$element, _, $points, $nodes>(
                        &saint_venant_kirchhoff(bulk_modulus),
                        distortion,
                        strain,
                    );
                    if strain <= 0.1 {
                        for (model, ratios) in [("neo-Hookean", neo), ("SVK", svk)] {
                            let median = ratios[ratios.len() / 2];
                            assert!(median < $loosest, "{model} {median}");
                        }
                    }
                }
            }
        }
    };
}

test_bounds!(tetrahedron, Tetrahedron<1>, 1, 4, 1.2);
test_bounds!(hexahedron, Hexahedron, 8, 8, 1.7);
test_bounds!(wedge, Wedge, 6, 6, 1.4);
test_bounds!(pyramid, Pyramid, 8, 5, 1.3);
test_bounds!(
    stabilized_hexahedron,
    stabilized::linear::Hexahedron,
    1,
    8,
    1.2
);

#[test]
fn the_hourglass_part_bounds_the_hourglass_stiffnesses() {
    let parametric: [[Scalar; 3]; 8] =
        from_fn(|a| from_fn(|i| Hexahedron::parametric_reference()[a][i].value()));
    for trial in 0..TRIALS {
        let (reference, _) = nodes(parametric, trial, 0.4, 0.0);
        let element: StabilizedElement<8> =
            stabilized::linear::Hexahedron::from(ElementNodalReferenceCoordinates::from(reference));
        let masses = element.nodal_lumped_masses(&[Density::kilograms_per_cubic_meter(1.0)].into());
        let shear_modulus = Stress::pascals(1.0);
        let exact = largest_eigenvalue(&element.hourglass_stiffnesses(shear_modulus), &masses);
        let bound = element
            .hourglass_squared_frequency_bound(shear_modulus, &masses)
            .value();
        assert!(exact > 0.0);
        assert!(bound >= exact * (1.0 - 1e-9), "{bound} is below {exact}");
        assert!(bound <= 2.0 * exact, "{bound} is far above {exact}");
    }
}

#[test]
fn a_stabilized_element_adds_the_hourglass_part_to_that_of_its_point() {
    let parametric: [[Scalar; 3]; 8] =
        from_fn(|a| from_fn(|i| Hexahedron::parametric_reference()[a][i].value()));
    let (reference, current) = nodes(parametric, 1, 0.2, 0.1);
    let element: StabilizedElement<8> =
        stabilized::linear::Hexahedron::from(ElementNodalReferenceCoordinates::from(reference));
    let coordinates = ElementNodalCoordinates::from(current);
    let masses = element.nodal_lumped_masses(&[Density::kilograms_per_cubic_meter(1.0)].into());
    let model = neo_hookean(1.0);
    let point =
        squared_frequency_bound_from(&element, &model, &ModelBound, &coordinates, &masses).unwrap();
    let whole = element
        .squared_frequency_bound(&model, &ModelBound, &coordinates, &masses)
        .unwrap();
    let hourglass = element.hourglass_squared_frequency_bound(model.shear_modulus, &masses);
    assert!(hourglass.value() > 0.0);
    assert_eq!(whole, point + hourglass);
}

#[test]
fn coordinates_that_are_not_numbers_are_an_error() {
    let reference = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ];
    let mut current = reference;
    current[2][1] = Scalar::NAN;
    let element = Tetrahedron::<1>::from(ElementNodalReferenceCoordinates::from(reference));
    let masses = element.nodal_lumped_masses(&[Density::kilograms_per_cubic_meter(1.0)].into());
    assert!(
        element
            .squared_frequency_bound(
                &neo_hookean(1.0),
                &ModelBound,
                &ElementNodalCoordinates::from(current),
                &masses,
            )
            .is_err()
    );
}
