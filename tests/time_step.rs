#![cfg(feature = "fem")]

use conspire::{
    constitutive::solid::hyperelastic::{NeoHookean, SaintVenantKirchhoff},
    fem::{
        Blocks, Model, NodalCoordinates, NodalReferenceCoordinates, NodalVelocities,
        block::Block,
        boundary::{BoundaryConditions, Solid},
        element::{linear::Hexahedron, stabilized},
        solid::{
            Checked, CourantFriedrichsLewy, ElasticExplicitDynamics, ElementPowerIteration,
            Execution, Lumped, TimeScaleEstimate, elastic::Workers,
        },
    },
    math::{
        Quantity, Tensor,
        integrate::{IntegrationError, VelocityVerlet},
    },
    units::{Density, Stress, Time},
};

type Soft = Block<NeoHookean, stabilized::linear::Hexahedron, 1, 3, 8, 8, Quantity<Density>>;
type Stiff = Block<SaintVenantKirchhoff, Hexahedron, 8, 3, 8, 8, Quantity<Density>>;
type Both = Blocks<Soft, Stiff>;

const CELLS: usize = 3;
const MANY_CELLS: usize = 6;

fn lattice(cells: usize) -> Vec<[f64; 3]> {
    (0..=cells)
        .flat_map(|k| {
            (0..=cells).flat_map(move |j| {
                (0..=cells).map(move |i| {
                    let [x, y, z] = [i, j, k].map(|index| index as f64 / cells as f64);
                    [x + 0.04 * y * z, y + 0.03 * x * x, z]
                })
            })
        })
        .collect()
}

fn layers(cells: usize, range: core::ops::Range<usize>) -> Vec<[usize; 8]> {
    let index = move |i: usize, j: usize, k: usize| (k * (cells + 1) + j) * (cells + 1) + i;
    range
        .flat_map(|k| {
            (0..cells).flat_map(move |j| {
                (0..cells).map(move |i| {
                    [
                        index(i, j, k),
                        index(i + 1, j, k),
                        index(i + 1, j + 1, k),
                        index(i, j + 1, k),
                        index(i, j, k + 1),
                        index(i + 1, j, k + 1),
                        index(i + 1, j + 1, k + 1),
                        index(i, j + 1, k + 1),
                    ]
                })
            })
        })
        .collect()
}

fn model() -> Model<Both, 3> {
    model_of(CELLS, Density::kilograms_per_cubic_meter(7.8e3))
}

fn model_of(cells: usize, density: Quantity<Density>) -> Model<Both, 3> {
    let reference = NodalReferenceCoordinates::from(lattice(cells));
    let soft = 2 * cells / 3;
    let blocks = Blocks(
        Soft::from((
            NeoHookean {
                shear_modulus: Stress::pascals(3.0e9),
                bulk_modulus: Stress::pascals(13.0e9),
            },
            density,
            layers(cells, 0..soft),
            &reference,
        )),
        Stiff::from((
            SaintVenantKirchhoff {
                shear_modulus: Stress::pascals(80.0e9),
                bulk_modulus: Stress::pascals(160.0e9),
            },
            density,
            layers(cells, soft..cells),
            &reference,
        )),
    );
    (blocks, reference).into()
}

fn at_rest() -> NodalCoordinates<3> {
    lattice(CELLS).into()
}

fn courant_friedrichs_lewy(model: &Model<Both, 3>) -> Quantity<Time> {
    CourantFriedrichsLewy::default()
        .time_scale(model, None, &at_rest())
        .unwrap()
}

fn run<E>(model: &Model<Both, 3>, checked: Checked<E>, step: f64) -> Result<(), IntegrationError>
where
    E: TimeScaleEstimate<Model<Both, 3>>,
{
    model
        .integrate(
            Execution::Serial,
            checked,
            Lumped,
            &VelocityVerlet::new(Time::seconds(step)),
            &[Time::seconds(0.0), Time::seconds(5.0 * step)],
            (
                at_rest(),
                NodalVelocities::from(vec![[0.0; 3]; lattice(CELLS).len()]),
            ),
            &(),
            &BoundaryConditions::<Solid<3>>::new(),
        )
        .map(|_| ())
}

#[test]
fn the_courant_friedrichs_lewy_time_scale_is_no_longer_than_that_of_the_elements() {
    let model = model();
    let bound = courant_friedrichs_lewy(&model).value();
    let elements = ElementPowerIteration
        .time_scale(&model, None, &at_rest())
        .unwrap()
        .value();
    assert!(bound <= elements, "{bound} is longer than {elements}");
    assert!(bound >= 0.6 * elements, "{bound} is far below {elements}");
}

#[test]
fn a_default_check_accepts_a_step_below_nine_tenths_of_the_limit_and_refuses_one_above() {
    let model = model();
    let limit = 0.9 * 2.0 * courant_friedrichs_lewy(&model).value();
    run(&model, Checked::default(), 0.99 * limit).unwrap();
    assert!(matches!(
        run(&model, Checked::default(), 1.01 * limit),
        Err(IntegrationError::UnstableTimeStep(..))
    ));
}

#[test]
fn a_step_that_the_elements_allow_may_be_refused_by_the_bound() {
    let model = model();
    let bound = courant_friedrichs_lewy(&model).value();
    let elements = ElementPowerIteration
        .time_scale(&model, None, &at_rest())
        .unwrap()
        .value();
    let step = bound + elements;
    let checked = Checked {
        estimate: ElementPowerIteration,
        interval: 1,
        safety: 1.0,
    };
    run(&model, checked, 0.999 * 2.0 * elements).unwrap();
    assert!(
        run(
            &model,
            Checked {
                estimate: CourantFriedrichsLewy::default(),
                interval: 1,
                safety: 1.0,
            },
            step
        )
        .is_err()
    );
}

#[test]
fn a_bound_that_is_not_finite_is_an_error() {
    let model = model_of(CELLS, Density::kilograms_per_cubic_meter(f64::NAN));
    assert!(
        CourantFriedrichsLewy::default()
            .time_scale(&model, None, &at_rest())
            .is_err()
    );
}

fn many() -> Model<Both, 3> {
    model_of(MANY_CELLS, Density::kilograms_per_cubic_meter(7.8e3))
}

fn stretched() -> Vec<[f64; 3]> {
    let mut coordinates = lattice(MANY_CELLS);
    let node = coordinates.len() - 10;
    coordinates[node][2] += 1.2 / MANY_CELLS as f64;
    coordinates
}

fn time_scales<E>(estimate: E, coordinates: Vec<[f64; 3]>)
where
    E: TimeScaleEstimate<Model<Both, 3>>,
{
    let model = many();
    let coordinates = NodalCoordinates::from(coordinates);
    let serial = estimate
        .time_scale(&model, None, &coordinates)
        .map_err(|error| error.to_string());
    for threads in [1, 2, 3, 7] {
        let workers = Workers::new(threads).unwrap();
        let parallel = estimate
            .time_scale(&model, Some(&workers), &coordinates)
            .map_err(|error| error.to_string());
        assert_eq!(parallel, serial, "{threads} threads");
    }
}

#[test]
fn any_number_of_threads_gives_the_serial_time_scales() {
    let model = many();
    let at_rest = NodalCoordinates::from(lattice(MANY_CELLS));
    let moved = NodalCoordinates::from(stretched());
    assert!(
        ElementPowerIteration
            .time_scale(&model, None, &moved)
            .unwrap()
            < ElementPowerIteration
                .time_scale(&model, None, &at_rest)
                .unwrap()
    );
    time_scales(CourantFriedrichsLewy::default(), stretched());
    time_scales(ElementPowerIteration, stretched());
}

#[test]
fn any_number_of_threads_gives_the_serial_error_of_an_estimate() {
    let mut inverted = lattice(MANY_CELLS);
    let nodes = inverted.len();
    inverted[nodes / 3][2] += 3.0;
    inverted[nodes - 10][2] -= 3.0;
    let model = many();
    assert!(
        CourantFriedrichsLewy::default()
            .time_scale(&model, None, &NodalCoordinates::from(inverted.clone()))
            .is_err()
    );
    time_scales(CourantFriedrichsLewy::default(), inverted.clone());
    time_scales(ElementPowerIteration, inverted);
}

#[test]
fn a_checked_run_on_threads_is_the_serial_run() {
    let model = many();
    let limit = 0.9
        * 2.0
        * CourantFriedrichsLewy::default()
            .time_scale(&model, None, &NodalCoordinates::from(stretched()))
            .unwrap()
            .value();
    let run = |execution, step: f64| {
        model
            .integrate(
                execution,
                Checked::default(),
                Lumped,
                &VelocityVerlet::new(Time::seconds(step)),
                &[Time::seconds(0.0), Time::seconds(25.0 * step)],
                (
                    stretched().into(),
                    NodalVelocities::from(vec![[0.0; 3]; stretched().len()]),
                ),
                &(),
                &BoundaryConditions::<Solid<3>>::new(),
            )
            .map(|(_, coordinates, _, _)| coordinates[coordinates.len() - 1].clone())
            .map_err(|error| format!("{error:?}"))
    };
    for step in [0.1 * limit, 1.01 * limit] {
        let serial = run(Execution::Serial, step);
        assert_eq!(serial.is_err(), step > limit);
        assert!(run(Execution::Threads(3), step) == serial);
    }
}
