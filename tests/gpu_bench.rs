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
//! `sparse_solves` times the sparse direct solve of the tangent of a stretched grid on the host and
//! on a device by cuDSS, and where the time of the device goes. `SOLVE_CELLS` is the cells on a
//! side, `8,16,24,32,48` by default, `HOST_MAX_DOFS` is the unknowns above which the host is not
//! run, 300000 by default, and one Newton solve of the largest grid the host is run on is timed on
//! both.
//!
//! The environment sets what is run: `CELLS` is the cells on a side of the grids of
//! `single_block`, `22,46,90` by default, `ELEMENTS` is the elements of each shape of
//! `element_shapes`, 200000 by default, `THREADS` is the threads, 4 by default, and
//! `REPETITIONS` is how many times each is timed, 5 by default.

mod common;

use common::{
    agree, components, deformed, density, describe, disjoint, lattice, on_device, rubber, slab,
    steel,
};
use conspire::{
    constitutive::solid::hyperelastic::{NeoHookean, SaintVenantKirchhoff},
    fem::{
        Blocks, Model, NodalCoordinates, NodalReferenceCoordinates, Root,
        block::Block,
        boundary::{BoundaryConditions, ConstrainAt, Displacement, Loads, Solid},
        element::linear::{Hexahedron, LinearElement},
        solid::{
            NodalForcesSolid,
            elastic::{ElasticElements, ParallelElasticElements, Workers},
        },
    },
    gpu::{Device, DeviceBlocks, DeviceSparseSolver, GpuError, Timings},
    math::{
        Hessian, Quantity, Tensor, Vector,
        assert::AssertionError,
        optimize::{NewtonRaphson, Tolerances},
        sparse::{SparseError, SparseSolver},
    },
    units::{Density, Length},
};
use std::{env::var, hint::black_box, str::FromStr, time::Instant};

type Grid = Block<NeoHookean, Hexahedron, 8, 3, 8, 8, Quantity<Density>>;
type Steel = Block<SaintVenantKirchhoff, Hexahedron, 8, 3, 8, 8, Quantity<Density>>;

fn variable<T: FromStr>(name: &str) -> Option<T> {
    var(name).ok().and_then(|value| value.parse().ok())
}

fn cells() -> Vec<usize> {
    var("CELLS")
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
    samples.sort_by(f64::total_cmp);
    samples[samples.len() / 2]
}

fn time_of<E>(repetitions: usize, mut operation: impl FnMut() -> Result<(), E>) -> Result<f64, E> {
    operation()?;
    let mut samples = Vec::with_capacity(repetitions);
    for _ in 0..repetitions {
        let start = Instant::now();
        operation()?;
        samples.push(start.elapsed().as_secs_f64() * 1e3);
    }
    Ok(median(samples))
}

fn elements<B: DeviceBlocks>(model: &Model<B, 3>) -> usize {
    let mut blocks = Vec::new();
    model.device_blocks(&mut blocks);
    blocks.iter().map(|block| block.layout.items).sum()
}

fn compare<B>(
    label: &str,
    model: &Model<B, 3>,
    current: &NodalCoordinates<3>,
    repetitions: usize,
) -> Result<(), AssertionError>
where
    B: ElasticElements<3> + ParallelElasticElements<3> + DeviceBlocks,
{
    let elements = elements(model);
    let per_element = |milliseconds: f64| milliseconds * 1e6 / elements as f64;
    let workers = Workers::new(threads()).map_err(describe)?;
    let mut scratch = model.scratch();
    let expected = components(&model.nodal_forces(current)?);
    let serial = time_of(repetitions, || {
        black_box(model.nodal_forces(black_box(current))?);
        Ok::<(), AssertionError>(())
    })?;
    let parallel = time_of(repetitions, || {
        let mut forces = NodalForcesSolid::<3>::zero(current.len());
        model.nodal_forces_parallel_into(
            &mut scratch,
            &workers,
            black_box(current),
            &mut forces,
        )?;
        black_box(forces);
        Ok::<(), AssertionError>(())
    })?;
    println!(
        "{label:>30}: {elements:>8} elements | serial {serial:9.3} ms {:7.1} ns/element | {} threads {parallel:9.3} ms {:7.1} ns/element",
        per_element(serial),
        threads(),
        per_element(parallel),
    );
    for fused in [false, true] {
        let start = Instant::now();
        let Some(resident) = on_device(model, fused) else {
            return Ok(());
        };
        let setup = start.elapsed().as_secs_f64() * 1e3;
        let found = resident.nodal_forces(model, current).map_err(describe)?;
        agree(&components(&found), &expected, 1e-10)?;
        let mut reported = Vec::new();
        for (name, resources) in resident.resources().map_err(describe)? {
            if !reported.contains(&name) {
                println!(
                    "{:>30}  {name}: {} registers, {} bytes of local memory, {} blocks of {} threads on a multiprocessor, {:.0}% of {} threads",
                    "",
                    resources.registers,
                    resources.local_bytes,
                    resources.blocks_per_multiprocessor,
                    resources.threads_per_block,
                    100.0 * resources.occupancy(),
                    resources.threads_per_multiprocessor,
                );
                reported.push(name);
            }
        }
        resident.set_coordinates(current).map_err(describe)?;
        let kernels = time_of(repetitions, || resident.run().map_err(describe))?;
        let transfers = time_of(repetitions, || {
            black_box(resident.nodal_forces(model, current).map_err(describe)?);
            Ok::<(), AssertionError>(())
        })?;
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
    Ok(())
}

fn grid(cells: usize) -> (Model<Grid, 3>, Vec<[f64; 3]>) {
    let coordinates = lattice(cells, cells);
    let reference = NodalReferenceCoordinates::from(coordinates.clone());
    let block = Grid::from((rubber(), density(), slab(cells, 0..cells), &reference));
    ((block, reference).into(), coordinates)
}

#[test]
#[ignore]
fn single_block() -> Result<(), AssertionError> {
    println!("\nthreads: {}", threads());
    for cells in cells() {
        let (model, coordinates) = grid(cells);
        let repetitions = (4_000_000 / cells.pow(3)).clamp(repetitions(), 200);
        compare(
            &format!("{} hexahedra", cells.pow(3)),
            &model,
            &deformed(&coordinates),
            repetitions,
        )?;
    }
    Ok(())
}

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
        )
    }};
}

#[test]
#[ignore]
fn element_shapes() -> Result<(), AssertionError> {
    println!("\nthreads: {}", threads());
    let count = variable("ELEMENTS").unwrap_or(200_000);
    shape!(count, "neo_hookean", NeoHookean, rubber(), 8, 8)?;
    shape!(count, "neo_hookean", NeoHookean, rubber(), 1, 4)?;
    shape!(count, "neo_hookean", NeoHookean, rubber(), 4, 4)?;
    shape!(count, "neo_hookean", NeoHookean, rubber(), 6, 6)?;
    shape!(count, "neo_hookean", NeoHookean, rubber(), 8, 5)?;
    shape!(
        count,
        "saint_venant_kirchhoff",
        SaintVenantKirchhoff,
        steel(),
        8,
        8
    )?;
    shape!(
        count,
        "saint_venant_kirchhoff",
        SaintVenantKirchhoff,
        steel(),
        1,
        4
    )?;
    shape!(
        count,
        "saint_venant_kirchhoff",
        SaintVenantKirchhoff,
        steel(),
        4,
        4
    )?;
    shape!(
        count,
        "saint_venant_kirchhoff",
        SaintVenantKirchhoff,
        steel(),
        6,
        6
    )?;
    shape!(
        count,
        "saint_venant_kirchhoff",
        SaintVenantKirchhoff,
        steel(),
        8,
        5
    )?;
    Ok(())
}

type Three = Blocks<Grid, Blocks<Steel, Grid>>;

const COLUMNS: usize = 32;

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
fn three_blocks_that_share_nodes() -> Result<(), AssertionError> {
    println!("\nthreads: {}", threads());
    let (model, coordinates) = three_blocks();
    compare(
        "three blocks",
        &model,
        &deformed(&coordinates),
        repetitions(),
    )
}

fn solve_cells() -> Vec<usize> {
    var("SOLVE_CELLS")
        .ok()
        .map(|cells| {
            cells
                .split(',')
                .filter_map(|cells| cells.parse().ok())
                .collect()
        })
        .unwrap_or_else(|| vec![8, 16, 24, 32, 48])
}

fn free_pattern(cells: usize) -> (Vec<(usize, usize)>, usize) {
    let face = (cells + 1) * (cells + 1);
    let unknowns = 3 * (face * (cells + 1) - face);
    let mut pattern = Vec::new();
    for element in slab(cells, 0..cells) {
        for &a in element.iter().filter(|&&a| a >= face) {
            for &b in element.iter().filter(|&&b| b >= face) {
                for (i, j) in (0..3).flat_map(|i| (0..3).map(move |j| (i, j))) {
                    pattern.push((3 * (a - face) + i, 3 * (b - face) + j));
                }
            }
        }
    }
    pattern.sort_unstable();
    pattern.dedup();
    (pattern, unknowns)
}

fn relative_residual(
    pattern: &[(usize, usize)],
    source: &impl Fn(usize, usize) -> f64,
    x: &Vector,
    b: &Vector,
) -> f64 {
    let mut product = vec![0.0; b.len()];
    pattern
        .iter()
        .for_each(|&(i, j)| product[i] += source(i, j) * x[j]);
    let worst = product
        .iter()
        .zip(b.iter())
        .fold(0.0_f64, |worst, (product, b)| {
            worst.max((product - b).abs())
        });
    worst / b.iter().fold(0.0_f64, |largest, b| largest.max(b.abs()))
}

fn milliseconds(seconds: f64) -> String {
    format!("{:.2}", 1e3 * seconds)
}

fn show_timings(label: &str, timings: &Timings) {
    println!(
        "    {label}: fill {} + gather {} + upload {} + analysis {} + factor {} + solve {} + download {} ms",
        milliseconds(timings.fill),
        milliseconds(timings.gather),
        milliseconds(timings.upload),
        milliseconds(timings.analysis),
        milliseconds(timings.factorization),
        milliseconds(timings.solve),
        milliseconds(timings.download),
    );
}

#[test]
#[ignore]
fn sparse_solves() -> Result<(), AssertionError> {
    let device = match Device::new(0) {
        Ok(device) => device,
        Err(GpuError::Unavailable(message)) => {
            println!("skipped: there is no device: {message}");
            return Ok(());
        }
        Err(error) => panic!("{error}"),
    };
    println!("\n{device:?}, the host solver is serial");
    let host_limit = variable("HOST_MAX_DOFS").unwrap_or(300_000);
    let mut last_with_host = None;
    for cells in solve_cells() {
        let (model, coordinates) = grid(cells);
        let tangent = model
            .nodal_stiffnesses(&deformed(&coordinates))
            .map_err(describe)?;
        let (pattern, unknowns) = free_pattern(cells);
        let offset = 3 * (cells + 1) * (cells + 1);
        let source = |i: usize, j: usize| tangent.entry(i + offset, j + offset);
        let b: Vector = (0..unknowns).map(|i| 1.0 + 0.1 * (i % 7) as f64).collect();
        let repetitions = (2_000_000 / unknowns).clamp(3, 20);
        println!(
            "\n{cells}^3 hexahedra: {unknowns} unknowns, {} entries",
            pattern.len()
        );
        let mut solver = DeviceSparseSolver::from_pattern(&device, unknowns, pattern.clone(), true)
            .map_err(describe)?;
        let start = Instant::now();
        let x = solver.solve(source, &b).map_err(describe)?;
        let first = start.elapsed().as_secs_f64() * 1e3;
        show_timings("device first", &solver.timings());
        let again = time_of(repetitions, || {
            black_box(solver.solve(source, &b)?);
            Ok::<(), SparseError>(())
        })
        .map_err(describe)?;
        show_timings("device again", &solver.timings());
        let device_residual = relative_residual(&pattern, &source, &x, &b);
        println!(
            "  device: first {first:.1} ms, then {again:.2} ms (median of {repetitions}), residual {device_residual:.1e}"
        );
        if unknowns > host_limit {
            println!("  host: not run above {host_limit} unknowns");
            continue;
        }
        let start = Instant::now();
        let host = SparseSolver::from_pattern(unknowns, pattern.clone(), true);
        let built = start.elapsed().as_secs_f64() * 1e3;
        let start = Instant::now();
        let x_host = host.solve(source, &b).map_err(describe)?;
        let first_host = start.elapsed().as_secs_f64() * 1e3;
        let again_host = time_of(repetitions, || {
            black_box(host.solve(source, &b)?);
            Ok::<(), SparseError>(())
        })
        .map_err(describe)?;
        let host_residual = relative_residual(&pattern, &source, &x_host, &b);
        println!(
            "  host: built {built:.1} ms, first {first_host:.1} ms, then {again_host:.2} ms, residual {host_residual:.1e}; device is {:.1}x on repeated solves",
            again_host / again
        );
        last_with_host = Some(cells);
    }
    if let Some(cells) = last_with_host {
        newton_on_both(&device, cells)?;
    }
    Ok(())
}

fn newton_on_both(device: &Device, cells: usize) -> Result<(), AssertionError> {
    let face = (cells + 1) * (cells + 1);
    let nodes = face * (cells + 1);
    let newton = |model: Model<Grid, 3>| {
        let bottom = 0..face;
        let conditions = BoundaryConditions::<Solid<3>>::new()
            .fix_nodes(Displacement::<0>, bottom.clone())
            .fix_nodes(Displacement::<1>, bottom.clone())
            .fix_nodes(Displacement::<2>, bottom)
            .prescribe_nodes(Displacement::<2>, nodes - face..nodes, Length::meters(0.01));
        let solver = NewtonRaphson {
            abs_tol: Tolerances {
                constraint: 1e-9,
                residual: 1e-3,
            },
            ..NewtonRaphson::default()
        };
        let start = Instant::now();
        let solved = model
            .root(&conditions, &Loads::<Solid<3>, ()>::new(), solver)
            .map_err(|error| error.to_string());
        (solved, start.elapsed().as_secs_f64())
    };
    let (host, host_time) = newton(grid(cells).0);
    let (found, device_time) = newton(grid(cells).0.with_device(device));
    let (host, found) = (host.map_err(describe)?, found.map_err(describe)?);
    let worst = host
        .iter()
        .zip(found.iter())
        .flat_map(|(host, found)| host.iter().zip(found.iter()))
        .fold(0.0_f64, |worst, (host, found)| {
            worst.max((host.value() - found.value()).abs())
        });
    println!(
        "\nA Newton solve of {cells}^3 hexahedra: host {:.2} s, device {:.2} s, the coordinates differ by {worst:.1e}",
        host_time, device_time
    );
    Ok(())
}
