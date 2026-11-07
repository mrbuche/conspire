#![cfg(feature = "gpu")]

//! Explicit dynamics on a CUDA device, against the same runs on the calling thread.
//!
//! These need a device, and pass without doing anything when there is none. They use only the
//! public interface, so that running them compiles the library and this file and not the unit
//! tests of the library: `cargo test -F gpu --test gpu -- --nocapture`.

use conspire::{
    constitutive::solid::hyperelastic::{NeoHookean, SaintVenantKirchhoff},
    fem::{
        Blocks, Model, NodalCoordinates, NodalReferenceCoordinates, NodalVelocities,
        block::{
            Block,
            element::linear::{Hexahedron, Pyramid, Tetrahedron, Wedge},
        },
        boundary::{BoundaryConditions, ConstrainAt, Displacement, LoadAt, Loads, Solid},
        solid::{
            DeviceExplicitDynamics, ElasticExplicitDynamics, Execution, GpuGradients, GpuKernels,
            GpuOptions, GpuPrecision, Lumped, NodalForcesSolid, Unchecked,
        },
    },
    math::{Quantity, Tensor, integrate::VelocityVerlet},
    units::{Density, Length, Stress, Time},
};

type Grid = Block<NeoHookean, Hexahedron, 8, 3, 8, 8, Quantity<Density>>;
type Elastic = Block<SaintVenantKirchhoff, Hexahedron, 8, 3, 8, 8, Quantity<Density>>;
type OnePoint = Block<NeoHookean, Tetrahedron<1>, 1, 3, 4, 4, Quantity<Density>>;
type FourPoints = Block<NeoHookean, Tetrahedron<4>, 4, 3, 4, 4, Quantity<Density>>;
type Prisms = Block<NeoHookean, Wedge, 6, 3, 6, 6, Quantity<Density>>;
type Pyramids = Block<NeoHookean, Pyramid, 8, 3, 5, 5, Quantity<Density>>;
type ElasticOnePoint = Block<SaintVenantKirchhoff, Tetrahedron<1>, 1, 3, 4, 4, Quantity<Density>>;
type ElasticPrisms = Block<SaintVenantKirchhoff, Wedge, 6, 3, 6, 6, Quantity<Density>>;

const CELLS: usize = 6;
const STEPS: usize = 40;
const STEP: f64 = 1e-8;

fn index(i: usize, j: usize, k: usize) -> usize {
    (k * (CELLS + 1) + j) * (CELLS + 1) + i
}

fn lattice() -> Vec<[f64; 3]> {
    (0..=CELLS)
        .flat_map(|k| {
            (0..=CELLS).flat_map(move |j| {
                (0..=CELLS).map(move |i| {
                    [
                        i as f64 / CELLS as f64,
                        j as f64 / CELLS as f64,
                        k as f64 / CELLS as f64,
                    ]
                })
            })
        })
        .collect()
}

fn hexahedra() -> Vec<[usize; 8]> {
    (0..CELLS)
        .flat_map(|k| {
            (0..CELLS).flat_map(move |j| {
                (0..CELLS).map(move |i| {
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

fn model() -> (Model<Grid, 3>, Vec<[f64; 3]>) {
    let coordinates = lattice();
    let reference = NodalReferenceCoordinates::from(coordinates.clone());
    (
        (
            Grid::from((
                NeoHookean {
                    shear_modulus: Stress::pascals(3.0e9),
                    bulk_modulus: Stress::pascals(13.0e9),
                },
                Density::kilograms_per_cubic_meter(7.8e3),
                hexahedra(),
                &reference,
            )),
            reference,
        )
            .into(),
        coordinates,
    )
}

/// A smooth displacement, so that every element has a stress that is not small.
fn deformed(coordinates: &[[f64; 3]]) -> NodalCoordinates<3> {
    coordinates
        .iter()
        .map(|&[x, y, z]| {
            [
                x + 2e-2 * (3.0 * y).sin() + 1e-2 * z,
                y + 2e-2 * (2.0 * z).sin(),
                z + 2e-2 * (4.0 * x).sin() - 1e-2 * x,
            ]
        })
        .collect::<Vec<_>>()
        .into()
}

fn velocities(coordinates: &[[f64; 3]]) -> NodalVelocities<3> {
    NodalVelocities::from(
        coordinates
            .iter()
            .map(|&[x, y, z]| [1.0 * y, -0.5 * z, 0.25 * x])
            .collect::<Vec<_>>(),
    )
}

fn components<T, U>(nodes: &T) -> Vec<f64>
where
    T: Tensor,
    T::Item: Tensor<Item = Quantity<U>>,
{
    nodes
        .iter()
        .flat_map(|node| node.iter().map(|component| component.value()))
        .collect()
}

/// The largest difference, and the largest of what it is a difference from.
fn difference(found: &[f64], expected: &[f64]) -> (f64, f64) {
    (
        found
            .iter()
            .zip(expected)
            .map(|(found, expected)| (found - expected).abs())
            .fold(0.0, f64::max),
        expected
            .iter()
            .map(|expected| expected.abs())
            .fold(0.0, f64::max),
    )
}

/// Whether an error is that there is no device, which these tests skip.
fn no_device(error: &impl std::fmt::Debug) -> bool {
    format!("{error:?}").contains("No CUDA device is available")
}

/// The run on the calling thread: the coordinates and the velocities at the end.
fn serial(
    model: &Model<Grid, 3>,
    coordinates: &NodalCoordinates<3>,
    velocities: &NodalVelocities<3>,
) -> (Vec<f64>, Vec<f64>) {
    let run = model
        .integrate(
            Execution::Serial,
            Unchecked,
            Lumped,
            &VelocityVerlet::new(Time::seconds(STEP)),
            &[Time::seconds(0.0), Time::seconds(STEPS as f64 * STEP)],
            (coordinates.clone(), velocities.clone()),
            &(),
            &BoundaryConditions::<Solid<3>>::new(),
        )
        .unwrap();
    (
        components(run.1.iter().last().unwrap()),
        components(run.2.iter().last().unwrap()),
    )
}

/// The run that copies the state at every step, as the integrator of the library does.
#[test]
fn run_with_history_follows_the_run_on_the_calling_thread() {
    let (model, reference) = model();
    let (coordinates, velocities) = (deformed(&reference), velocities(&reference));
    let expected = serial(&model, &coordinates, &velocities);
    let run = model.integrate(
        Execution::Gpu(GpuOptions::default()),
        Unchecked,
        Lumped,
        &VelocityVerlet::new(Time::seconds(STEP)),
        &[Time::seconds(0.0), Time::seconds(STEPS as f64 * STEP)],
        (coordinates.clone(), velocities.clone()),
        &(),
        &BoundaryConditions::<Solid<3>>::new(),
    );
    let run = match run {
        Ok(run) => run,
        Err(error) if no_device(&error) => return println!("skipped: {error:?}"),
        Err(error) => panic!("{error:?}"),
    };
    let (gap, scale) = difference(&components(run.1.iter().last().unwrap()), &expected.0);
    assert!(
        gap < 1e-12 * scale.max(1.0),
        "the coordinates differ by {gap}"
    );
    let (gap, scale) = difference(&components(run.2.iter().last().unwrap()), &expected.1);
    assert!(
        gap < 1e-9 * scale.max(1.0),
        "the velocities differ by {gap}"
    );
}

/// The run that stays on the device, in both precisions.
#[test]
fn run_on_the_device_follows_the_run_on_the_calling_thread() {
    let (model, reference) = model();
    let (coordinates, velocities) = (deformed(&reference), velocities(&reference));
    let expected = serial(&model, &coordinates, &velocities);
    for (precision, tolerance) in [(GpuPrecision::F64, 1e-9), (GpuPrecision::F32, 1e-3)] {
        let run = model.integrate_on_device(
            GpuOptions {
                precision,
                ..GpuOptions::default()
            },
            Time::seconds(STEP),
            STEPS,
            (coordinates.clone(), velocities.clone()),
            &(),
            &BoundaryConditions::<Solid<3>>::new(),
        );
        let (found_coordinates, found_velocities) = match run {
            Ok(run) => run,
            Err(error) if no_device(&error) => return println!("skipped: {error:?}"),
            Err(error) => panic!("{error:?}"),
        };
        let (gap, scale) = difference(&components(&found_coordinates), &expected.0);
        assert!(
            gap < tolerance * scale.max(1.0),
            "{precision:?}: the coordinates differ by {gap}"
        );
        let (gap, scale) = difference(&components(&found_velocities), &expected.1);
        assert!(
            gap < tolerance * scale.max(1.0),
            "{precision:?}: the velocities differ by {gap}"
        );
    }
}

/// The run that stays on the device, with the code of the library in place of the kernels written
/// for it: the forces of the elements, the sums on the nodes and the steps, in double precision.
#[test]
fn run_with_the_code_of_the_library_follows_the_run_on_the_calling_thread() {
    let (model, reference) = model();
    let (coordinates, velocities) = (deformed(&reference), velocities(&reference));
    let expected = serial(&model, &coordinates, &velocities);
    for gradients in [GpuGradients::Stored, GpuGradients::OnTheFly] {
        let run = model.integrate_on_device(
            GpuOptions {
                kernels: GpuKernels::Library,
                gradients,
                ..GpuOptions::default()
            },
            Time::seconds(STEP),
            STEPS,
            (coordinates.clone(), velocities.clone()),
            &(),
            &BoundaryConditions::<Solid<3>>::new(),
        );
        let (found_coordinates, found_velocities) = match run {
            Ok(run) => run,
            Err(error) if no_device(&error) => return println!("skipped: {error:?}"),
            Err(error) => panic!("{error:?}"),
        };
        let (gap, scale) = difference(&components(&found_coordinates), &expected.0);
        assert!(
            gap < 1e-9 * scale.max(1.0),
            "{gradients:?}: the coordinates differ by {gap}"
        );
        let (gap, scale) = difference(&components(&found_velocities), &expected.1);
        assert!(
            gap < 1e-9 * scale.max(1.0),
            "{gradients:?}: the velocities differ by {gap}"
        );
    }
}

/// A lattice twice as long in one direction, and the cells on each side of the plane in the middle
/// of it, which the nodes of the two sides share.
/// The coordinates of the nodes and the nodes of the cells of each of two blocks.
type TwoLattices = (Vec<[f64; 3]>, Vec<[usize; 8]>, Vec<[usize; 8]>);

fn two_lattices() -> TwoLattices {
    let nodes_x = 2 * CELLS + 1;
    let index = move |i: usize, j: usize, k: usize| (k * (CELLS + 1) + j) * nodes_x + i;
    let coordinates = (0..=CELLS)
        .flat_map(|k| {
            (0..=CELLS).flat_map(move |j| {
                (0..nodes_x).map(move |i| {
                    [
                        i as f64 / CELLS as f64,
                        j as f64 / CELLS as f64,
                        k as f64 / CELLS as f64,
                    ]
                })
            })
        })
        .collect::<Vec<_>>();
    let cells = |start: usize| {
        (0..CELLS)
            .flat_map(|k| {
                (0..CELLS).flat_map(move |j| {
                    (start..start + CELLS).map(move |i| {
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
            .collect::<Vec<_>>()
    };
    (coordinates, cells(0), cells(CELLS))
}

/// Two blocks of different materials that share the plane of nodes between them.
fn two_blocks() -> (Model<Blocks<Grid, Grid>, 3>, Vec<[f64; 3]>) {
    let (coordinates, first_cells, second_cells) = two_lattices();
    let reference = NodalReferenceCoordinates::from(coordinates.clone());
    let first = Grid::from((
        NeoHookean {
            shear_modulus: Stress::pascals(3.0e9),
            bulk_modulus: Stress::pascals(13.0e9),
        },
        Density::kilograms_per_cubic_meter(7.8e3),
        first_cells,
        &reference,
    ));
    let second = Grid::from((
        NeoHookean {
            shear_modulus: Stress::pascals(1.0e9),
            bulk_modulus: Stress::pascals(5.0e9),
        },
        Density::kilograms_per_cubic_meter(2.7e3),
        second_cells,
        &reference,
    ));
    ((Blocks(first, second), reference).into(), coordinates)
}

/// Two blocks of different models, one Neo-Hookean and one Saint Venant-Kirchhoff, that share the
/// plane of nodes between them.
fn mixed_blocks() -> (Model<Blocks<Grid, Elastic>, 3>, Vec<[f64; 3]>) {
    let (coordinates, first_cells, second_cells) = two_lattices();
    let reference = NodalReferenceCoordinates::from(coordinates.clone());
    let first = Grid::from((
        NeoHookean {
            shear_modulus: Stress::pascals(3.0e9),
            bulk_modulus: Stress::pascals(13.0e9),
        },
        Density::kilograms_per_cubic_meter(7.8e3),
        first_cells,
        &reference,
    ));
    let second = Elastic::from((
        SaintVenantKirchhoff {
            shear_modulus: Stress::pascals(1.0e9),
            bulk_modulus: Stress::pascals(5.0e9),
        },
        Density::kilograms_per_cubic_meter(2.7e3),
        second_cells,
        &reference,
    ));
    ((Blocks(first, second), reference).into(), coordinates)
}

/// A model of two blocks, which have different materials and share nodes, on the device in every
/// way that there is, against the run on the calling thread.
#[test]
fn run_of_a_model_of_two_blocks_follows_the_run_on_the_calling_thread() {
    let (model, reference) = two_blocks();
    let (coordinates, velocities) = (deformed(&reference), velocities(&reference));
    let expected = model
        .integrate(
            Execution::Serial,
            Unchecked,
            Lumped,
            &VelocityVerlet::new(Time::seconds(STEP)),
            &[Time::seconds(0.0), Time::seconds(STEPS as f64 * STEP)],
            (coordinates.clone(), velocities.clone()),
            &(),
            &BoundaryConditions::<Solid<3>>::new(),
        )
        .unwrap();
    let expected = (
        components(expected.1.iter().last().unwrap()),
        components(expected.2.iter().last().unwrap()),
    );
    for (precision, gradients, kernels, tolerance) in [
        (
            GpuPrecision::F64,
            GpuGradients::Stored,
            GpuKernels::Handwritten,
            1e-9,
        ),
        (
            GpuPrecision::F64,
            GpuGradients::OnTheFly,
            GpuKernels::Handwritten,
            1e-9,
        ),
        (
            GpuPrecision::F32,
            GpuGradients::Stored,
            GpuKernels::Handwritten,
            1e-3,
        ),
        (
            GpuPrecision::F64,
            GpuGradients::Stored,
            GpuKernels::Library,
            1e-9,
        ),
        (
            GpuPrecision::F64,
            GpuGradients::OnTheFly,
            GpuKernels::Library,
            1e-9,
        ),
    ] {
        let run = model.integrate_on_device(
            GpuOptions {
                precision,
                gradients,
                kernels,
                ..GpuOptions::default()
            },
            Time::seconds(STEP),
            STEPS,
            (coordinates.clone(), velocities.clone()),
            &(),
            &BoundaryConditions::<Solid<3>>::new(),
        );
        let (found_coordinates, found_velocities) = match run {
            Ok(run) => run,
            Err(error) if no_device(&error) => return println!("skipped: {error:?}"),
            Err(error) => panic!("{precision:?} {gradients:?} {kernels:?}: {error:?}"),
        };
        let (gap, scale) = difference(&components(&found_coordinates), &expected.0);
        assert!(
            gap < tolerance * scale.max(1.0),
            "{precision:?} {gradients:?} {kernels:?}: the coordinates differ by {gap}"
        );
        let (gap, scale) = difference(&components(&found_velocities), &expected.1);
        assert!(
            gap < tolerance * scale.max(1.0),
            "{precision:?} {gradients:?} {kernels:?}: the velocities differ by {gap}"
        );
    }
}

/// A model of two blocks of different models, which share nodes, with the code of the library on
/// the device, against the run on the calling thread. The kernels that are written for the device
/// are for the Neo-Hookean model alone, so they are to say that they cannot take it.
#[test]
fn run_of_a_model_of_two_models_follows_the_run_on_the_calling_thread() {
    let (model, reference) = mixed_blocks();
    let (coordinates, velocities) = (deformed(&reference), velocities(&reference));
    let expected = model
        .integrate(
            Execution::Serial,
            Unchecked,
            Lumped,
            &VelocityVerlet::new(Time::seconds(STEP)),
            &[Time::seconds(0.0), Time::seconds(STEPS as f64 * STEP)],
            (coordinates.clone(), velocities.clone()),
            &(),
            &BoundaryConditions::<Solid<3>>::new(),
        )
        .unwrap();
    let expected = (
        components(expected.1.iter().last().unwrap()),
        components(expected.2.iter().last().unwrap()),
    );
    for gradients in [GpuGradients::Stored, GpuGradients::OnTheFly] {
        let run = model.integrate_on_device(
            GpuOptions {
                kernels: GpuKernels::Library,
                gradients,
                ..GpuOptions::default()
            },
            Time::seconds(STEP),
            STEPS,
            (coordinates.clone(), velocities.clone()),
            &(),
            &BoundaryConditions::<Solid<3>>::new(),
        );
        let (found_coordinates, found_velocities) = match run {
            Ok(run) => run,
            Err(error) if no_device(&error) => return println!("skipped: {error:?}"),
            Err(error) => panic!("{gradients:?}: {error:?}"),
        };
        let (gap, scale) = difference(&components(&found_coordinates), &expected.0);
        assert!(
            gap < 1e-9 * scale.max(1.0),
            "{gradients:?}: the coordinates differ by {gap}"
        );
        let (gap, scale) = difference(&components(&found_velocities), &expected.1);
        assert!(
            gap < 1e-9 * scale.max(1.0),
            "{gradients:?}: the velocities differ by {gap}"
        );
    }
    let refused = model.integrate_on_device(
        GpuOptions::default(),
        Time::seconds(STEP),
        STEPS,
        (coordinates, velocities),
        &(),
        &BoundaryConditions::<Solid<3>>::new(),
    );
    let error = refused.expect_err("the kernels written for the device are for one model");
    assert!(
        format!("{error:?}").contains("Neo-Hookean"),
        "the error is not that: {error:?}"
    );
}

// The linear elements other than the hexahedron, cut from the cells of the lattice of the grid.
//
// The corners of a cell are numbered by their bits, `1` for the first axis, `2` for the second
// and `4` for the third, so that the cell is `corner(c)` for `c` in `0..8`. The nodes of an
// element are ordered as the element asks, which is checked from the volume that they make: the
// elements of the library are on the side of the corners that gives them a positive Jacobian.

/// The sign of the volume of the tetrahedron that four points make, in the order of the element.
fn volume(coordinates: &[[f64; 3]], a: usize, b: usize, c: usize, d: usize) -> f64 {
    let subtract = |x: usize, y: usize| {
        [
            coordinates[x][0] - coordinates[y][0],
            coordinates[x][1] - coordinates[y][1],
            coordinates[x][2] - coordinates[y][2],
        ]
    };
    let (u, v, w) = (subtract(b, a), subtract(c, a), subtract(d, a));
    (u[1] * v[2] - u[2] * v[1]) * w[0]
        + (u[2] * v[0] - u[0] * v[2]) * w[1]
        + (u[0] * v[1] - u[1] * v[0]) * w[2]
}

/// The elements of every kind in the cells of the grid whose first index is in `slab`, the nodes
/// being those of the lattice and the center of each cell that has pyramids.
#[derive(Default)]
struct Cut {
    tetrahedra: Vec<[usize; 4]>,
    wedges: Vec<[usize; 6]>,
    pyramids: Vec<[usize; 5]>,
}

/// The lattice, with the centers of the cells of `pyramids` after its nodes, and the elements.
fn cut_lattice(
    tetrahedra: std::ops::Range<usize>,
    wedges: std::ops::Range<usize>,
    pyramids: std::ops::Range<usize>,
) -> (Vec<[f64; 3]>, Cut) {
    let mut coordinates = lattice();
    let mut cut = Cut::default();
    for k in 0..CELLS {
        for j in 0..CELLS {
            for i in 0..CELLS {
                let corner = |c: usize| index(i + (c & 1), j + ((c >> 1) & 1), k + (c >> 2));
                if tetrahedra.contains(&i) {
                    // The six paths along the edges of the cell from one corner to the opposite.
                    for axes in [
                        [1, 2, 4],
                        [1, 4, 2],
                        [2, 1, 4],
                        [2, 4, 1],
                        [4, 1, 2],
                        [4, 2, 1],
                    ] {
                        let mut nodes = [
                            corner(0),
                            corner(axes[0]),
                            corner(axes[0] | axes[1]),
                            corner(7),
                        ];
                        if volume(&coordinates, nodes[0], nodes[1], nodes[2], nodes[3]) < 0.0 {
                            nodes.swap(1, 2);
                        }
                        cut.tetrahedra.push(nodes);
                    }
                }
                if wedges.contains(&i) {
                    for corners in [[0, 1, 2, 4, 5, 6], [3, 2, 1, 7, 6, 5]] {
                        let mut nodes = corners.map(corner);
                        if volume(&coordinates, nodes[0], nodes[1], nodes[2], nodes[3]) < 0.0 {
                            nodes.swap(1, 2);
                            nodes.swap(4, 5);
                        }
                        cut.wedges.push(nodes);
                    }
                }
                if pyramids.contains(&i) {
                    let center = coordinates.len();
                    let mut middle = [0.0; 3];
                    (0..8).for_each(|c| {
                        (0..3).for_each(|x| middle[x] += coordinates[corner(c)][x] / 8.0)
                    });
                    coordinates.push(middle);
                    // The six faces of the cell, each the base of a pyramid on the center.
                    for base in [
                        [0, 1, 3, 2],
                        [4, 5, 7, 6],
                        [0, 1, 5, 4],
                        [2, 3, 7, 6],
                        [0, 2, 6, 4],
                        [1, 3, 7, 5],
                    ] {
                        let mut nodes = [
                            corner(base[0]),
                            corner(base[1]),
                            corner(base[2]),
                            corner(base[3]),
                            center,
                        ];
                        if volume(&coordinates, nodes[0], nodes[1], nodes[3], nodes[4]) < 0.0 {
                            nodes.swap(1, 3);
                        }
                        cut.pyramids.push(nodes);
                    }
                }
            }
        }
    }
    (coordinates, cut)
}

/// The ways of running a model on a device, with the tolerance of each: every way that there is
/// for a model that is all Neo-Hookean, and the library alone for any.
const EVERY_WAY: [(GpuPrecision, GpuGradients, GpuKernels, f64); 5] = [
    (
        GpuPrecision::F64,
        GpuGradients::Stored,
        GpuKernels::Handwritten,
        1e-9,
    ),
    (
        GpuPrecision::F64,
        GpuGradients::OnTheFly,
        GpuKernels::Handwritten,
        1e-9,
    ),
    (
        GpuPrecision::F32,
        GpuGradients::Stored,
        GpuKernels::Handwritten,
        1e-3,
    ),
    (
        GpuPrecision::F64,
        GpuGradients::Stored,
        GpuKernels::Library,
        1e-9,
    ),
    (
        GpuPrecision::F64,
        GpuGradients::OnTheFly,
        GpuKernels::Library,
        1e-9,
    ),
];
const LIBRARY_WAYS: [(GpuPrecision, GpuGradients, GpuKernels, f64); 2] = [
    (
        GpuPrecision::F64,
        GpuGradients::Stored,
        GpuKernels::Library,
        1e-9,
    ),
    (
        GpuPrecision::F64,
        GpuGradients::OnTheFly,
        GpuKernels::Library,
        1e-9,
    ),
];

/// Checks that a model, which is given with the coordinates of its nodes, runs on a device in each
/// of the ways as it does on the calling thread. A macro, so that the model can be of any blocks.
macro_rules! follows {
    ($model:expr, $reference:expr, $ways:expr, $what:expr) => {{
        let (model, reference) = (&$model, &$reference);
        let (coordinates, velocities) = (deformed(reference), velocities(reference));
        let expected = model
            .integrate(
                Execution::Serial,
                Unchecked,
                Lumped,
                &VelocityVerlet::new(Time::seconds(STEP)),
                &[Time::seconds(0.0), Time::seconds(STEPS as f64 * STEP)],
                (coordinates.clone(), velocities.clone()),
                &(),
                &BoundaryConditions::<Solid<3>>::new(),
            )
            .unwrap();
        let expected = (
            components(expected.1.iter().last().unwrap()),
            components(expected.2.iter().last().unwrap()),
        );
        for (precision, gradients, kernels, tolerance) in $ways {
            let run = model.integrate_on_device(
                GpuOptions {
                    precision,
                    gradients,
                    kernels,
                    ..GpuOptions::default()
                },
                Time::seconds(STEP),
                STEPS,
                (coordinates.clone(), velocities.clone()),
                &(),
                &BoundaryConditions::<Solid<3>>::new(),
            );
            let (found_coordinates, found_velocities) = match run {
                Ok(run) => run,
                Err(error) if no_device(&error) => return println!("skipped: {error:?}"),
                Err(error) => panic!(
                    "{} {precision:?} {gradients:?} {kernels:?}: {error:?}",
                    $what
                ),
            };
            let (gap, scale) = difference(&components(&found_coordinates), &expected.0);
            assert!(
                gap < tolerance * scale.max(1.0),
                "{} {precision:?} {gradients:?} {kernels:?}: the coordinates differ by {gap}",
                $what
            );
            let (gap, scale) = difference(&components(&found_velocities), &expected.1);
            assert!(
                gap < tolerance * scale.max(1.0),
                "{} {precision:?} {gradients:?} {kernels:?}: the velocities differ by {gap}",
                $what
            );
        }
    }};
}

fn soft() -> NeoHookean {
    NeoHookean {
        shear_modulus: Stress::pascals(1.0e9),
        bulk_modulus: Stress::pascals(5.0e9),
    }
}

fn stiff() -> NeoHookean {
    NeoHookean {
        shear_modulus: Stress::pascals(3.0e9),
        bulk_modulus: Stress::pascals(13.0e9),
    }
}

fn elastic() -> SaintVenantKirchhoff {
    SaintVenantKirchhoff {
        shear_modulus: Stress::pascals(2.0e9),
        bulk_modulus: Stress::pascals(9.0e9),
    }
}

#[test]
fn run_of_tetrahedra_of_one_point_follows_the_run_on_the_calling_thread() {
    let (coordinates, cut) = cut_lattice(0..CELLS, 0..0, 0..0);
    let reference = NodalReferenceCoordinates::from(coordinates.clone());
    let block = OnePoint::from((
        stiff(),
        Density::kilograms_per_cubic_meter(7.8e3),
        cut.tetrahedra,
        &reference,
    ));
    let model: Model<OnePoint, 3> = (block, reference).into();
    follows!(model, coordinates, EVERY_WAY, "tetrahedra of one point");
}

#[test]
fn run_of_tetrahedra_of_four_points_follows_the_run_on_the_calling_thread() {
    let (coordinates, cut) = cut_lattice(0..CELLS, 0..0, 0..0);
    let reference = NodalReferenceCoordinates::from(coordinates.clone());
    let block = FourPoints::from((
        stiff(),
        Density::kilograms_per_cubic_meter(7.8e3),
        cut.tetrahedra,
        &reference,
    ));
    let model: Model<FourPoints, 3> = (block, reference).into();
    follows!(model, coordinates, EVERY_WAY, "tetrahedra of four points");
}

#[test]
fn run_of_wedges_follows_the_run_on_the_calling_thread() {
    let (coordinates, cut) = cut_lattice(0..0, 0..CELLS, 0..0);
    let reference = NodalReferenceCoordinates::from(coordinates.clone());
    let block = Prisms::from((
        stiff(),
        Density::kilograms_per_cubic_meter(7.8e3),
        cut.wedges,
        &reference,
    ));
    let model: Model<Prisms, 3> = (block, reference).into();
    follows!(model, coordinates, EVERY_WAY, "wedges");
}

#[test]
fn run_of_pyramids_follows_the_run_on_the_calling_thread() {
    let (coordinates, cut) = cut_lattice(0..0, 0..0, 0..CELLS);
    let reference = NodalReferenceCoordinates::from(coordinates.clone());
    let block = Pyramids::from((
        stiff(),
        Density::kilograms_per_cubic_meter(7.8e3),
        cut.pyramids,
        &reference,
    ));
    let model: Model<Pyramids, 3> = (block, reference).into();
    follows!(model, coordinates, EVERY_WAY, "pyramids");
}

/// Five blocks of five kinds of elements, in slabs of the grid that share the nodes of the planes
/// between them: hexahedra, tetrahedra of one point and of four, wedges and pyramids.
#[test]
fn run_of_a_model_of_every_kind_of_element_follows_the_run_on_the_calling_thread() {
    let (coordinates, cut) = cut_lattice(2..3, 4..5, 5..6);
    // The tetrahedra of one point are the first half of them and those of four the second.
    let reference = NodalReferenceCoordinates::from(coordinates.clone());
    let (first, second): (Vec<_>, Vec<_>) = {
        let half = cut.tetrahedra.len() / 2;
        let mut all = cut.tetrahedra;
        let second = all.split_off(half);
        (all, second)
    };
    let slab = |cells: std::ops::Range<usize>| {
        hexahedra()
            .into_iter()
            .enumerate()
            .filter(|(cell, _)| cells.contains(&(cell % CELLS)))
            .map(|(_, nodes)| nodes)
            .collect::<Vec<_>>()
    };
    let blocks = Blocks(
        Grid::from((
            stiff(),
            Density::kilograms_per_cubic_meter(7.8e3),
            slab(0..2),
            &reference,
        )),
        Blocks(
            OnePoint::from((
                soft(),
                Density::kilograms_per_cubic_meter(2.7e3),
                first,
                &reference,
            )),
            Blocks(
                FourPoints::from((
                    stiff(),
                    Density::kilograms_per_cubic_meter(7.8e3),
                    second,
                    &reference,
                )),
                Blocks(
                    Prisms::from((
                        soft(),
                        Density::kilograms_per_cubic_meter(2.7e3),
                        cut.wedges,
                        &reference,
                    )),
                    Pyramids::from((
                        stiff(),
                        Density::kilograms_per_cubic_meter(7.8e3),
                        cut.pyramids,
                        &reference,
                    )),
                ),
            ),
        ),
    );
    let model: Model<_, 3> = (blocks, reference).into();
    follows!(model, coordinates, EVERY_WAY, "five kinds of elements");
}

/// The same, with a different model in some of the blocks, which the library alone can run.
#[test]
fn run_of_a_model_of_every_kind_of_element_and_two_models_follows_the_run_on_the_calling_thread() {
    let (coordinates, cut) = cut_lattice(2..4, 4..5, 5..6);
    let reference = NodalReferenceCoordinates::from(coordinates.clone());
    let slab = |cells: std::ops::Range<usize>| {
        hexahedra()
            .into_iter()
            .enumerate()
            .filter(|(cell, _)| cells.contains(&(cell % CELLS)))
            .map(|(_, nodes)| nodes)
            .collect::<Vec<_>>()
    };
    let blocks = Blocks(
        Grid::from((
            stiff(),
            Density::kilograms_per_cubic_meter(7.8e3),
            slab(0..2),
            &reference,
        )),
        Blocks(
            ElasticOnePoint::from((
                elastic(),
                Density::kilograms_per_cubic_meter(2.7e3),
                cut.tetrahedra,
                &reference,
            )),
            Blocks(
                ElasticPrisms::from((
                    elastic(),
                    Density::kilograms_per_cubic_meter(2.7e3),
                    cut.wedges,
                    &reference,
                )),
                Pyramids::from((
                    soft(),
                    Density::kilograms_per_cubic_meter(7.8e3),
                    cut.pyramids,
                    &reference,
                )),
            ),
        ),
    );
    let model: Model<_, 3> = (blocks, reference).into();
    follows!(
        model,
        coordinates,
        LIBRARY_WAYS,
        "five kinds of elements, two models"
    );
}

/// A run in chunks of the elements of a block gives the results of the run in one chunk, to the
/// bit, since each node is given the forces of its elements in the order of the elements either
/// way.
#[test]
fn run_in_chunks_of_elements_gives_the_results_of_the_run_in_one_chunk() {
    let (model, reference) = two_blocks();
    let (coordinates, velocities) = (deformed(&reference), velocities(&reference));
    let run = |gradients, chunk| {
        model.integrate_on_device(
            GpuOptions {
                kernels: GpuKernels::Library,
                gradients,
                chunk,
                ..GpuOptions::default()
            },
            Time::seconds(STEP),
            STEPS,
            (coordinates.clone(), velocities.clone()),
            &(),
            &BoundaryConditions::<Solid<3>>::new(),
        )
    };
    for gradients in [GpuGradients::Stored, GpuGradients::OnTheFly] {
        let whole = match run(gradients, 0) {
            Ok(whole) => (components(&whole.0), components(&whole.1)),
            Err(error) if no_device(&error) => return println!("skipped: {error:?}"),
            Err(error) => panic!("{gradients:?}: {error:?}"),
        };
        for chunk in [1, 7, 100, 215, 216, 1000] {
            let found = run(gradients, chunk).unwrap();
            assert!(
                components(&found.0) == whole.0,
                "{gradients:?}, chunks of {chunk}: the coordinates are not those of one chunk"
            );
            assert!(
                components(&found.1) == whole.1,
                "{gradients:?}, chunks of {chunk}: the velocities are not those of one chunk"
            );
        }
    }
}

/// One face of the grid is held and the opposite one is pulled on a ramp, which moves it along the
/// axis at a rate that is not that of the velocities it starts with, and a device holds it where
/// the calling thread does at each step, in every way that there is.
#[test]
fn run_with_a_face_that_is_moved_follows_the_run_on_the_calling_thread() {
    let (model, reference) = model();
    let (coordinates, velocities) = (deformed(&reference), velocities(&reference));
    let face = |i: usize| {
        (0..=CELLS)
            .flat_map(|k| (0..=CELLS).map(move |j| index(i, j, k)))
            .collect::<Vec<_>>()
    };
    let (held, pulled) = (face(0), face(CELLS));
    let duration = STEPS as f64 * STEP;
    let conditions = BoundaryConditions::<Solid<3>>::new()
        .fix_nodes(Displacement::<0>, held.clone())
        .fix_nodes(Displacement::<1>, held.clone())
        .fix_nodes(Displacement::<2>, held)
        .prescribe_nodes_with(Displacement::<0>, pulled, move |time| {
            Length::meters(2e-3 * time.value() / duration)
        });
    let expected = model
        .integrate(
            Execution::Serial,
            Unchecked,
            Lumped,
            &VelocityVerlet::new(Time::seconds(STEP)),
            &[Time::seconds(0.0), Time::seconds(duration)],
            (coordinates.clone(), velocities.clone()),
            &(),
            &conditions,
        )
        .unwrap();
    let expected = (
        components(expected.1.iter().last().unwrap()),
        components(expected.2.iter().last().unwrap()),
    );
    // The face moved, so that the run is not the run of a body that is held.
    let moved = difference(&expected.0, &components(&coordinates)).0;
    assert!(moved > 1e-3, "the face did not move: {moved}");
    for (precision, gradients, kernels, tolerance) in EVERY_WAY {
        let run = model.integrate_on_device(
            GpuOptions {
                precision,
                gradients,
                kernels,
                ..GpuOptions::default()
            },
            Time::seconds(STEP),
            STEPS,
            (coordinates.clone(), velocities.clone()),
            &(),
            &conditions,
        );
        let (found_coordinates, found_velocities) = match run {
            Ok(run) => run,
            Err(error) if no_device(&error) => return println!("skipped: {error:?}"),
            Err(error) => panic!("{precision:?} {gradients:?} {kernels:?}: {error:?}"),
        };
        let (gap, scale) = difference(&components(&found_coordinates), &expected.0);
        assert!(
            gap < tolerance * scale.max(1.0),
            "{precision:?} {gradients:?} {kernels:?}: the coordinates differ by {gap}"
        );
        let (gap, scale) = difference(&components(&found_velocities), &expected.1);
        assert!(
            gap < tolerance * scale.max(1.0),
            "{precision:?} {gradients:?} {kernels:?}: the velocities differ by {gap}"
        );
    }
}

/// Forces that are constant and forces that are a function of time are subtracted from those of the
/// elements as they are on the calling thread, in every way that there is.
#[test]
fn run_with_loads_follows_the_run_on_the_calling_thread() {
    let (model, reference) = model();
    let (coordinates, velocities) = (deformed(&reference), velocities(&reference));
    let nodes = NodalReferenceCoordinates::from(reference.clone());
    let duration = STEPS as f64 * STEP;
    let loads = Loads::<Solid<3>>::new()
        .nodal_forces(
            &nodes,
            NodalForcesSolid::from(
                (0..reference.len())
                    .map(|node| {
                        [
                            1e6 * ((node % 7) as f64 - 3.0),
                            5e5 * ((node % 5) as f64 - 2.0),
                            -2e6,
                        ]
                    })
                    .collect::<Vec<_>>(),
            ),
        )
        .unwrap()
        .body_force_with(&nodes, &model.nodal_volumes(), move |time| {
            let fraction = time.value() / duration;
            [1e9 * (6.0 * fraction).sin(), -5e8 * fraction, 2e9].into()
        })
        .unwrap();
    let conditions = BoundaryConditions::<Solid<3>>::new();
    let serial = |with_loads: bool| {
        let time = [Time::seconds(0.0), Time::seconds(duration)];
        let integrator = VelocityVerlet::new(Time::seconds(STEP));
        let state = (coordinates.clone(), velocities.clone());
        let run = if with_loads {
            model.integrate(
                Execution::Serial,
                Unchecked,
                Lumped,
                &integrator,
                &time,
                state,
                &loads,
                &conditions,
            )
        } else {
            model.integrate(
                Execution::Serial,
                Unchecked,
                Lumped,
                &integrator,
                &time,
                state,
                &(),
                &conditions,
            )
        }
        .unwrap();
        (
            components(run.1.iter().last().unwrap()),
            components(run.2.iter().last().unwrap()),
        )
    };
    let (expected, free) = (serial(true), serial(false));
    // The loads do something, so that a device that ignored them would be found out.
    let effect = difference(&expected.1, &free.1).0;
    assert!(effect > 1e-3, "the loads did not change the run: {effect}");
    for (precision, gradients, kernels, tolerance) in EVERY_WAY {
        let run = model.integrate_on_device(
            GpuOptions {
                precision,
                gradients,
                kernels,
                ..GpuOptions::default()
            },
            Time::seconds(STEP),
            STEPS,
            (coordinates.clone(), velocities.clone()),
            &loads,
            &conditions,
        );
        let (found_coordinates, found_velocities) = match run {
            Ok(run) => run,
            Err(error) if no_device(&error) => return println!("skipped: {error:?}"),
            Err(error) => panic!("{precision:?} {gradients:?} {kernels:?}: {error:?}"),
        };
        let (gap, scale) = difference(&components(&found_coordinates), &expected.0);
        assert!(
            gap < tolerance * scale.max(1.0),
            "{precision:?} {gradients:?} {kernels:?}: the coordinates differ by {gap}"
        );
        let (gap, scale) = difference(&components(&found_velocities), &expected.1);
        assert!(
            gap < tolerance * scale.max(1.0),
            "{precision:?} {gradients:?} {kernels:?}: the velocities differ by {gap}"
        );
    }
}

/// A load that follows the body, as a pressure does, cannot be tabulated before the run, so it is
/// refused before anything is put on a device, which a load that is not a function of the state
/// says that it is not.
#[test]
fn loads_that_depend_on_the_state_are_refused() {
    struct Follower;
    impl LoadAt<NodalCoordinates<3>, NodalForcesSolid<3>> for Follower {
        fn subtract_at(
            &self,
            _: Quantity<Time>,
            _: &NodalCoordinates<3>,
            _: &mut NodalForcesSolid<3>,
        ) -> Result<(), String> {
            Ok(())
        }
    }
    let (model, reference) = model();
    let error = model
        .integrate_on_device(
            GpuOptions::default(),
            Time::seconds(STEP),
            STEPS,
            (deformed(&reference), velocities(&reference)),
            &Follower,
            &BoundaryConditions::<Solid<3>>::new(),
        )
        .expect_err("the load depends on the state");
    assert!(
        format!("{error:?}").contains("depend on the state"),
        "the error is not that: {error:?}"
    );
}
