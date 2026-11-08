#![cfg(all(feature = "gpu", feature = "fem"))]

//! The nodal forces of a model on the calling thread, on threads, and on a CUDA device.
//!
//! On a device two things are timed. The kernels are the forces with the coordinates already on
//! the device, which is what a run that keeps its state there pays for each evaluation. The
//! transfers are the whole call: the coordinates up, the kernels, and the forces down. The setup
//! is what is done once: loading the kernels, which the driver compiles for the device, and
//! copying the blocks. Every case checks the forces of the device against the host before it is
//! timed, and is timed with the operations of the kernels not fused and fused.
//!
//! Run alone, one test at a time, in release mode, so that only the library and this file are
//! compiled and nothing else competes for the cores or the device:
//! `cargo test --release -F "gpu fem" --test gpu_bench -- --ignored --nocapture --test-threads=1`
//!
//! The environment sets what is run: `CELLS` is the cells on a side of the grids of
//! `single_block`, `22,46,90` by default, `ELEMENTS` is the elements of each shape of
//! `element_shapes`, 200000 by default, `THREADS` is the threads, 4 by default, `DEVICE` is the
//! ordinal of the device, and `REPETITIONS` is how many times each is timed, 5 by default.

mod common;

use common::{
    agree, components, deformed, density, disjoint, lattice, on_device, rubber, slab, steel,
};
use conspire::{
    constitutive::solid::hyperelastic::{NeoHookean, SaintVenantKirchhoff},
    fem::{
        Blocks, Model, NodalCoordinates, NodalReferenceCoordinates,
        block::Block,
        element::linear::{Hexahedron, LinearElement},
        solid::{
            NodalForcesSolid,
            elastic::{ElasticElements, ParallelElasticElements, Workers},
        },
    },
    gpu::DeviceBlocks,
    math::{Quantity, Tensor},
    units::Density,
};
use std::{hint::black_box, str::FromStr, time::Instant};

type Grid = Block<NeoHookean, Hexahedron, 8, 3, 8, 8, Quantity<Density>>;
type Steel = Block<SaintVenantKirchhoff, Hexahedron, 8, 3, 8, 8, Quantity<Density>>;

fn variable<T: FromStr>(name: &str) -> Option<T> {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
}

fn cells() -> Vec<usize> {
    std::env::var("CELLS")
        .ok()
        .map(|cells| {
            cells
                .split(',')
                .filter_map(|cells| cells.parse().ok())
                .collect()
        })
        .unwrap_or_else(|| vec![22, 46, 90])
}

fn threads() -> usize {
    variable("THREADS").unwrap_or(4)
}

fn repetitions() -> usize {
    variable("REPETITIONS").unwrap_or(5)
}

fn median(mut samples: Vec<f64>) -> f64 {
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    samples[samples.len() / 2]
}

/// The median time of an operation in milliseconds, after one call that is not timed.
fn time_of(repetitions: usize, mut operation: impl FnMut()) -> f64 {
    operation();
    median(
        (0..repetitions)
            .map(|_| {
                let start = Instant::now();
                operation();
                start.elapsed().as_secs_f64() * 1e3
            })
            .collect(),
    )
}

/// The number of elements of a model, which the blocks it is described by have.
fn elements<B: DeviceBlocks>(model: &Model<B, 3>) -> usize {
    let mut blocks = Vec::new();
    model.device_blocks(&mut blocks);
    blocks.iter().map(|block| block.layout.items).sum()
}

/// Times the forces of a model on the calling thread, on threads, and on the device.
fn compare<B>(label: &str, model: &Model<B, 3>, current: &NodalCoordinates<3>, repetitions: usize)
where
    B: ElasticElements<3> + ParallelElasticElements<3> + DeviceBlocks,
{
    let elements = elements(model);
    let per_element = |milliseconds: f64| milliseconds * 1e6 / elements as f64;
    let workers = Workers::new(threads()).expect("the threads");
    let mut scratch = model.scratch();
    let expected = components(&model.nodal_forces(current).expect("the forces on the host"));
    let serial = time_of(repetitions, || {
        black_box(model.nodal_forces(black_box(current)).expect("the forces"));
    });
    let parallel = time_of(repetitions, || {
        let mut forces = NodalForcesSolid::<3>::zero(current.len());
        model
            .nodal_forces_parallel_into(&mut scratch, &workers, black_box(current), &mut forces)
            .expect("the forces");
        black_box(forces);
    });
    println!(
        "{label:>30}: {elements:>8} elements | serial {serial:9.3} ms {:7.1} ns/element | {} threads {parallel:9.3} ms {:7.1} ns/element",
        per_element(serial),
        threads(),
        per_element(parallel),
    );
    for fused in [false, true] {
        let start = Instant::now();
        let Some(resident) = on_device(model, fused) else {
            return;
        };
        let setup = start.elapsed().as_secs_f64() * 1e3;
        let found = resident
            .nodal_forces(model, current)
            .expect("the forces on the device");
        agree(&components(&found), &expected, 1e-10);
        resident.set_coordinates(current).expect("the coordinates");
        let kernels = time_of(repetitions, || resident.run().expect("the forces"));
        let transfers = time_of(repetitions, || {
            black_box(resident.nodal_forces(model, current).expect("the forces"));
        });
        println!(
            "{:>30}  {} {:>7}: kernels {kernels:9.3} ms {:7.2} ns/element | with transfers {transfers:9.3} ms | setup {setup:8.1} ms | {:5.1}x serial {:5.1}x threads",
            "",
            resident.device_name(),
            if fused { "fused" } else { "unfused" },
            per_element(kernels),
            serial / kernels,
            parallel / kernels,
        );
    }
}

/// A grid of hexahedra of a rubber, of `cells` cells on a side.
fn grid(cells: usize) -> (Model<Grid, 3>, Vec<[f64; 3]>) {
    let coordinates = lattice(cells, cells);
    let reference = NodalReferenceCoordinates::from(coordinates.clone());
    let block = Grid::from((rubber(), density(), slab(cells, 0..cells), &reference));
    ((block, reference).into(), coordinates)
}

#[test]
#[ignore]
fn single_block() {
    println!("\nthreads: {}", threads());
    for cells in cells() {
        let (model, coordinates) = grid(cells);
        let repetitions = (4_000_000 / cells.pow(3)).clamp(repetitions(), 200);
        compare(
            &format!("{} hexahedra", cells.pow(3)),
            &model,
            &deformed(&coordinates),
            repetitions,
        );
    }
}

/// Blocks of each shape of element and each model, of elements that share no nodes, so that the
/// cost of an element is not mixed with that of the nodes.
macro_rules! shape {
    ($count:expr, $label:literal, $constitutive:ty, $model:expr, $points:literal, $nodes:literal) => {{
        type Shape = Block<
            $constitutive,
            LinearElement<$points, $nodes>,
            $points,
            3,
            $nodes,
            $nodes,
            Quantity<Density>,
        >;
        let (connectivity, coordinates) = disjoint::<$points, $nodes>($count);
        let reference = NodalReferenceCoordinates::from(coordinates.clone());
        let model: Model<Shape, 3> = (
            Shape::from(($model, density(), connectivity, &reference)),
            reference,
        )
            .into();
        compare(
            concat!($label, " g", $points, " n", $nodes),
            &model,
            &deformed(&coordinates),
            repetitions(),
        );
    }};
}

#[test]
#[ignore]
fn element_shapes() {
    println!("\nthreads: {}", threads());
    let count = variable("ELEMENTS").unwrap_or(200_000);
    shape!(count, "neo_hookean", NeoHookean, rubber(), 8, 8);
    shape!(count, "neo_hookean", NeoHookean, rubber(), 1, 4);
    shape!(count, "neo_hookean", NeoHookean, rubber(), 4, 4);
    shape!(count, "neo_hookean", NeoHookean, rubber(), 6, 6);
    shape!(count, "neo_hookean", NeoHookean, rubber(), 8, 5);
    shape!(
        count,
        "saint_venant_kirchhoff",
        SaintVenantKirchhoff,
        steel(),
        8,
        8
    );
    shape!(
        count,
        "saint_venant_kirchhoff",
        SaintVenantKirchhoff,
        steel(),
        1,
        4
    );
    shape!(
        count,
        "saint_venant_kirchhoff",
        SaintVenantKirchhoff,
        steel(),
        4,
        4
    );
    shape!(
        count,
        "saint_venant_kirchhoff",
        SaintVenantKirchhoff,
        steel(),
        6,
        6
    );
    shape!(
        count,
        "saint_venant_kirchhoff",
        SaintVenantKirchhoff,
        steel(),
        8,
        5
    );
}

type Three = Blocks<Grid, Blocks<Steel, Grid>>;

const COLUMNS: usize = 32;

/// Three slabs of about 100k, 30k and 1k elements, with different constitutive models, that share
/// the nodes on their interfaces.
fn three_blocks() -> (Model<Three, 3>, Vec<[f64; 3]>) {
    let (first, second) = (98, 29);
    let layers = first + second + 1;
    let coordinates = lattice(COLUMNS, layers);
    let reference = NodalReferenceCoordinates::from(coordinates.clone());
    let blocks = Blocks(
        Grid::from((rubber(), density(), slab(COLUMNS, 0..first), &reference)),
        Blocks(
            Steel::from((
                steel(),
                density(),
                slab(COLUMNS, first..first + second),
                &reference,
            )),
            Grid::from((
                rubber(),
                density(),
                slab(COLUMNS, first + second..layers),
                &reference,
            )),
        ),
    );
    ((blocks, reference).into(), coordinates)
}

#[test]
#[ignore]
fn three_blocks_that_share_nodes() {
    println!("\nthreads: {}", threads());
    let (model, coordinates) = three_blocks();
    compare(
        "three blocks",
        &model,
        &deformed(&coordinates),
        repetitions(),
    );
}
