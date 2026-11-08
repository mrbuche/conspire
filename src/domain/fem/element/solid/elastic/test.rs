use super::{ElasticElement, ElasticFiniteElement};
use crate::{
    constitutive::{ConstitutiveError, solid::hyperelastic::NeoHookean},
    fem::element::{
        ElementNodalCoordinates, ElementNodalReferenceCoordinates, IntegrationData,
        linear::Hexahedron,
    },
    math::{Quantity, Reference, Tensor, TensorRank1List},
    units::{ReciprocalLength, Stress, Volume},
};

const G: usize = 8;
const N: usize = 8;

type Forces = <Hexahedron as ElasticElement<NeoHookean>>::Forces;

fn model() -> NeoHookean {
    NeoHookean {
        bulk_modulus: Stress::pascals(13.0e9),
        shear_modulus: Stress::pascals(3.0e9),
    }
}

const CUBE: [[f64; 3]; N] = [
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [1.0, 1.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
    [1.0, 0.0, 1.0],
    [1.0, 1.0, 1.0],
    [0.0, 1.0, 1.0],
];

fn reference_coordinates(size: f64) -> ElementNodalReferenceCoordinates<N> {
    CUBE.map(|[x, y, z]| [size * x, size * y, size * z]).into()
}

fn current_coordinates(size: f64) -> ElementNodalCoordinates<N> {
    CUBE.map(|[x, y, z]| {
        let (x, y, z) = (size * x, size * y, size * z);
        [
            x + 0.05 * y * z + 0.02 * (3.0 * y).sin(),
            y + 0.04 * z * x + 0.02 * (2.0 * z).sin(),
            z + 0.03 * x * y + 0.02 * (4.0 * x).sin(),
        ]
    })
    .into()
}

fn bits(forces: &Forces) -> Vec<u64> {
    forces
        .iter()
        .flat_map(|force| force.iter().map(|component| component.value().to_bits()))
        .collect()
}

struct ElementLast<'a> {
    values: &'a [f64],
    items: usize,
    item: usize,
}

impl IntegrationData<G, N> for ElementLast<'_> {
    fn gradient_vectors_at(
        &self,
        point: usize,
    ) -> TensorRank1List<3, Reference, N, ReciprocalLength> {
        let mut gradient_vectors = [[0.0; 3]; N];
        for (node, gradient_vector) in gradient_vectors.iter_mut().enumerate() {
            for (component, value) in gradient_vector.iter_mut().enumerate() {
                *value = self.values[((point * N + node) * 3 + component) * self.items + self.item];
            }
        }
        gradient_vectors.into()
    }
    fn integration_weight_at(&self, point: usize) -> Quantity<Volume> {
        Volume::cubic_meters(self.values[(G * N * 3 + point) * self.items + self.item])
    }
}

#[test]
fn the_forces_from_data_with_the_element_last_are_the_forces_of_the_element() {
    let sizes = [1.0, 0.7, 1.3];
    let elements = sizes.map(|size| Hexahedron::from(reference_coordinates(size)));
    let items = elements.len();
    let mut values = vec![0.0; (G * N * 3 + G) * items];
    elements.iter().enumerate().for_each(|(item, element)| {
        for point in 0..G {
            let gradient_vectors = element.gradient_vectors_at(point);
            for (node, gradient_vector) in gradient_vectors.iter().enumerate() {
                for (component, value) in gradient_vector.iter().enumerate() {
                    values[((point * N + node) * 3 + component) * items + item] = value.value();
                }
            }
            values[(G * N * 3 + point) * items + item] =
                element.integration_weight_at(point).value();
        }
    });
    let model = model();
    for (item, (element, size)) in elements.iter().zip(sizes).enumerate() {
        let coordinates = current_coordinates(size);
        let found =
            <Hexahedron as ElasticFiniteElement<NeoHookean, G, 3, N, N>>::nodal_forces_from(
                &ElementLast {
                    values: &values,
                    items,
                    item,
                },
                &model,
                &coordinates,
            )
            .unwrap();
        let expected = element.nodal_forces(&model, &coordinates).unwrap();
        assert_eq!(bits(&found), bits(&expected), "element {item}");
    }
}

struct Inverted<'a> {
    element: &'a Hexahedron,
    point: usize,
}

impl IntegrationData<G, N> for Inverted<'_> {
    fn gradient_vectors_at(
        &self,
        point: usize,
    ) -> TensorRank1List<3, Reference, N, ReciprocalLength> {
        let gradient_vectors = self.element.gradient_vectors_at(point);
        if point == self.point {
            gradient_vectors * -1.0
        } else {
            gradient_vectors
        }
    }
    fn integration_weight_at(&self, point: usize) -> Quantity<Volume> {
        self.element.integration_weight_at(point)
    }
}

#[test]
fn a_stress_that_fails_at_an_integration_point_is_reported_at_that_point() {
    let element = Hexahedron::from(reference_coordinates(1.0));
    let model = model();
    let coordinates = current_coordinates(1.0);
    for point in [0, 3, G - 1] {
        let error =
            <Hexahedron as ElasticFiniteElement<NeoHookean, G, 3, N, N>>::nodal_forces_from(
                &Inverted {
                    element: &element,
                    point,
                },
                &model,
                &coordinates,
            )
            .expect_err("the stress at the point fails");
        assert_eq!(error.point, Some(point));
        assert!(matches!(
            error.source,
            ConstitutiveError::InvalidJacobian(jacobian) if jacobian < 0.0
        ));
    }
}

#[test]
fn both_ways_to_make_the_forces_report_the_first_integration_point_that_fails() {
    let element = Hexahedron::from(reference_coordinates(1.0));
    let model = model();
    let inverted: ElementNodalCoordinates<N> = CUBE.map(|[x, y, z]| [-x, y, z]).into();
    let from_list = element.nodal_forces(&model, &inverted).unwrap_err();
    let from_data =
        <Hexahedron as ElasticFiniteElement<NeoHookean, G, 3, N, N>>::nodal_forces_from(
            &element, &model, &inverted,
        )
        .unwrap_err();
    assert_eq!(from_list.point, Some(0));
    assert_eq!(from_data.point, from_list.point);
    assert!(from_list == from_data);
}
