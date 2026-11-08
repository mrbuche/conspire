use super::OnTheFly;
use crate::{
    constitutive::solid::hyperelastic::NeoHookean,
    fem::element::{
        ElementNodalReferenceCoordinates, FiniteElement, IntegrationData,
        linear::{LinearElement, LinearFiniteElement},
        solid::elastic::{ElasticElement, ElasticFiniteElement},
    },
    math::Tensor,
    units::Stress,
};
use core::array::from_fn;

fn reference<const G: usize, const N: usize>() -> [[f64; 3]; N]
where
    LinearElement<G, N>: FiniteElement<G, 3, N, N>,
{
    let affine = [[1.1, 0.2, -0.1], [0.05, 0.9, 0.15], [-0.1, 0.12, 1.2]];
    let shift = [0.3, -0.2, 0.5];
    let mut nodes = [[0.0; 3]; N];
    <LinearElement<G, N> as FiniteElement<G, 3, N, N>>::parametric_reference()
        .iter()
        .zip(nodes.iter_mut())
        .enumerate()
        .for_each(|(a, (parametric, node))| {
            node.iter_mut().enumerate().for_each(|(i, coordinate)| {
                *coordinate = shift[i]
                    + (0..3)
                        .map(|k| affine[i][k] * parametric[k].value())
                        .sum::<f64>()
                    + 0.03 * (1.7 * a as f64 + i as f64).sin();
            })
        });
    nodes
}

fn is_the_element<const G: usize, const N: usize>()
where
    LinearElement<G, N>: LinearFiniteElement<G, N> + ElasticFiniteElement<NeoHookean, G, 3, N, N>,
{
    let nodes = reference::<G, N>();
    let element = LinearElement::<G, N>::from(ElementNodalReferenceCoordinates::<N>::from(nodes));
    let (derivatives, weights) = OnTheFly::<G, N>::tables::<LinearElement<G, N>>();
    let on_the_fly = OnTheFly::new(nodes, &derivatives, &weights);
    let (mut largest, mut worst) = (0.0_f64, 0.0_f64);
    for point in 0..G {
        let (expected, found) = (
            element.gradient_vectors_at(point),
            on_the_fly.gradient_vectors_at(point),
        );
        for (expected, found) in expected.iter().zip(found.iter()) {
            for (expected, found) in expected.iter().zip(found.iter()) {
                largest = largest.max(expected.value().abs());
                worst = worst.max((expected.value() - found.value()).abs());
            }
        }
        let (expected, found) = (
            element.integration_weight_at(point).value(),
            on_the_fly.integration_weight_at(point).value(),
        );
        largest = largest.max(expected.abs());
        worst = worst.max((expected - found).abs());
    }
    assert!(worst > 0.0 || largest > 0.0);
    assert!(
        worst <= 1e-12 * largest,
        "the data differ by {worst} of up to {largest}"
    );
    let current = from_fn(|a| {
        from_fn(|i| {
            nodes[a][i]
                + 0.03 * (2.3 * a as f64 + 1.1 * i as f64).sin()
                + 0.02 * nodes[a][(i + 1) % 3]
        })
    })
    .into();
    let model = NeoHookean {
        shear_modulus: Stress::pascals(3.0e9),
        bulk_modulus: Stress::pascals(13.0e9),
    };
    let expected = <LinearElement<G, N> as ElasticElement<NeoHookean>>::nodal_forces(
        &element, &model, &current,
    )
    .unwrap();
    let found =
        <LinearElement<G, N> as ElasticFiniteElement<NeoHookean, G, 3, N, N>>::nodal_forces_from(
            &on_the_fly,
            &model,
            &current,
        )
        .unwrap();
    let (mut largest, mut worst) = (0.0_f64, 0.0_f64);
    for (expected, found) in expected.iter().zip(found.iter()) {
        for (expected, found) in expected.iter().zip(found.iter()) {
            largest = largest.max(expected.value().abs());
            worst = worst.max((expected.value() - found.value()).abs());
        }
    }
    assert!(largest > 1e3, "the forces are {largest}, which is nothing");
    assert!(
        worst <= 1e-10 * largest,
        "the forces differ by {worst} of up to {largest}"
    );
}

#[test]
fn hexahedra_made_on_the_fly_are_the_elements() {
    is_the_element::<8, 8>();
}

#[test]
fn tetrahedra_of_one_point_made_on_the_fly_are_the_elements() {
    is_the_element::<1, 4>();
}

#[test]
fn tetrahedra_of_four_points_made_on_the_fly_are_the_elements() {
    is_the_element::<4, 4>();
}

#[test]
fn wedges_made_on_the_fly_are_the_elements() {
    is_the_element::<6, 6>();
}

#[test]
fn pyramids_made_on_the_fly_are_the_elements() {
    is_the_element::<8, 5>();
}
