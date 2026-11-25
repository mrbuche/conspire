use super::Recomputed;
use crate::{
    constitutive::solid::hyperelastic::NeoHookean,
    fem::element::{
        ElementNodalReferenceCoordinates, FiniteElement, IntegrationData,
        linear::{LinearElement, LinearFiniteElement},
        solid::elastic::{ElasticElement, ElasticFiniteElement},
    },
    math::{
        Tensor,
        assert::{Assert, AssertionError},
    },
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

fn is_the_element<const G: usize, const N: usize>() -> Result<(), AssertionError>
where
    LinearElement<G, N>: LinearFiniteElement<G, N> + ElasticFiniteElement<NeoHookean, G, 3, N, N>,
{
    let nodes = reference::<G, N>();
    let element = LinearElement::from(ElementNodalReferenceCoordinates::from(nodes));
    let (derivatives, weights) = Recomputed::tables::<LinearElement<G, N>>();
    let recomputed = Recomputed::new(nodes.into(), &derivatives, &weights);
    (0..G).try_for_each(|point| {
        Assert::eq(
            recomputed.gradient_vectors_at(point),
            element.gradient_vectors_at(point),
        )?;
        Assert::eq(
            recomputed.integration_weight_at(point).value(),
            element.integration_weight_at(point).value(),
        )
    })?;
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
    )?;
    let found =
        <LinearElement<G, N> as ElasticFiniteElement<NeoHookean, G, 3, N, N>>::nodal_forces_from(
            &recomputed,
            &model,
            &current,
        )?;
    assert!(
        expected
            .iter()
            .any(|force| force.iter().any(|component| component.value().abs() > 1e3)),
        "the forces are nothing"
    );
    Assert::eq(&found, &expected)
}

#[test]
fn hexahedra_recomputed_are_the_elements() -> Result<(), AssertionError> {
    is_the_element::<8, 8>()
}

#[test]
fn tetrahedra_of_one_point_recomputed_are_the_elements() -> Result<(), AssertionError> {
    is_the_element::<1, 4>()
}

#[test]
fn tetrahedra_of_four_points_recomputed_are_the_elements() -> Result<(), AssertionError> {
    is_the_element::<4, 4>()
}

#[test]
fn wedges_recomputed_are_the_elements() -> Result<(), AssertionError> {
    is_the_element::<6, 6>()
}

#[test]
fn pyramids_recomputed_are_the_elements() -> Result<(), AssertionError> {
    is_the_element::<8, 5>()
}
