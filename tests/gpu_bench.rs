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
//! `element_shapes`, 200000 by default, `THREADS` is the threads, 4 by default, and
//! `REPETITIONS` is how many times each is timed, 5 by default.
//!
//! `steps` times whole steps of velocity Verlet on a grid of hexahedra, without loads, with loads,
//! and with loads and a moved top: on the calling thread and on threads, with the integrator of
//! the library, and on a device with the state kept there. The cost of a step on the host is the
//! difference between a long run and a short one over the difference in their steps, which leaves
//! out what a run does once and is how the integrator that keeps a history is timed; on a device
//! it is the time of a long run of steps from when the run is started, so the setup, which is
//! large, is left out and reported. The forces alone are timed next to it, so that the step less
//! the forces is what the update and the launches cost. `CELLS` for `steps` is `10,22,46,90` by
//! default, and `HOST=0` leaves out the host, which is slow on the large grids.

mod common;

use common::{
    RATE, Work, agree, close, components, deformed, density, describe, disjoint, flat, imposed_by,
    lattice, on_device, rubber, schedule_of, slab, steel, targets_at, time_at, volume_loads_on,
};
use conspire::{
    constitutive::solid::hyperelastic::{NeoHookean, SaintVenantKirchhoff},
    fem::{
        Blocks, Model, NodalCoordinates, NodalReferenceCoordinates,
        block::Block,
        boundary::{BoundaryConditions, ConstrainAt, Displacement, LoadAt, Solid},
        element::{
            linear::{Hexahedron, LinearElement},
            stabilized::linear::Hexahedron as StabilizedHexahedron,
        },
        solid::{
            ElasticExplicitDynamics, Execution, Lumped, MassChoice, NodalForcesSolid, Unchecked,
            elastic::{ElasticElements, ParallelElasticElements, Workers},
            mass::LumpedMassElements,
        },
    },
    gpu::{DeviceBlocks, Resident},
    math::{Quantity, Tensor, assert::AssertionError, integrate::VelocityVerlet},
    units::{Density, Length, Time},
};
use std::{
    env::{var, var_os},
    hint::black_box,
    str::FromStr,
    time::Instant,
};

type Grid = Block<NeoHookean, Hexahedron, 8, 3, 8, 8, Quantity<Density>>;
type Steel = Block<SaintVenantKirchhoff, Hexahedron, 8, 3, 8, 8, Quantity<Density>>;
type StabilizedGrid = Block<NeoHookean, StabilizedHexahedron, 1, 3, 8, 8, Quantity<Density>>;

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
        let (model, coordinates) = stabilized_grid(cells);
        compare(
            &format!("{} stabilized", cells.pow(3)),
            &model,
            &deformed(&coordinates),
            repetitions,
        )?;
    }
    Ok(())
}

fn stabilized_grid(cells: usize) -> (Model<StabilizedGrid, 3>, Vec<[f64; 3]>) {
    let coordinates = lattice(cells, cells);
    let reference = NodalReferenceCoordinates::from(coordinates.clone());
    let block = StabilizedGrid::from((rubber(), density(), slab(cells, 0..cells), &reference));
    ((block, reference).into(), coordinates)
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

macro_rules! stabilized {
    ($count:expr, $label:literal, $constitutive:ty, $model:expr) => {{
        type Shape = Block<$constitutive, StabilizedHexahedron, 1, 3, 8, 8, Quantity<Density>>;
        let (connectivity, coordinates) = disjoint::<8, 8>($count);
        let reference = NodalReferenceCoordinates::from(coordinates.clone());
        let model: Model<Shape, 3> = (
            Shape::from(($model, density(), connectivity, &reference)),
            reference,
        )
            .into();
        compare(
            concat!($label, " stabilized n8"),
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
    stabilized!(count, "neo_hookean", NeoHookean, rubber())?;
    stabilized!(
        count,
        "saint_venant_kirchhoff",
        SaintVenantKirchhoff,
        steel()
    )?;
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

// The step of the runs of dynamics, which is exact in binary and small enough that the grids stay
// stable and the elements stay valid however many steps a device takes.
const STEP: f64 = 1.0 / 134_217_728.0;

// The most steps that a device takes in a timed run, so that the motion of the top and the loads
// of time stay small next to the cells of the smallest grids.
const MOST_STEPS: usize = 100_000;

fn step_cells() -> Vec<usize> {
    if var_os("CELLS").is_some() {
        cells()
    } else {
        vec![10, 22, 46, 90]
    }
}

// The steps of the runs on the host that keep a history, which is large, so fewer for larger
// grids.
fn history_steps(nodes: usize) -> (usize, usize) {
    let short = (400_000 / nodes).clamp(1, 20);
    (short, 3 * short + 1)
}

// The cost of a step in milliseconds, from runs of `short` and `long` steps.
fn marginal(
    short: usize,
    long: usize,
    mut run: impl FnMut(usize) -> Result<(), AssertionError>,
) -> Result<f64, AssertionError> {
    let short_time = time_of(repetitions(), || run(short))?;
    let long_time = time_of(repetitions(), || run(long))?;
    Ok((long_time - short_time) / (long - short) as f64)
}

// The coordinates, velocities and accelerations at the end of `steps` steps with the integrator
// of the library, each as `3 * node + component`.
fn on_host<B, L>(
    work: &Work<B>,
    loads: &L,
    moved: bool,
    execution: Execution,
    steps: usize,
) -> Result<[Vec<f64>; 3], AssertionError>
where
    B: ParallelElasticElements<3> + LumpedMassElements<3>,
    L: LoadAt<NodalCoordinates<3>, NodalForcesSolid<3>>,
{
    let top = if moved { work.top.clone() } else { Vec::new() };
    let base = 0..work.layer;
    let conditions = BoundaryConditions::<Solid<3>>::new()
        .fix_nodes(Displacement::<0>, base.clone())
        .fix_nodes(Displacement::<1>, base.clone())
        .fix_nodes(Displacement::<2>, base)
        .prescribe_nodes_with(Displacement::<0>, top, |time| {
            Length::meters(RATE * time.value())
        });
    let time = (0..=steps)
        .map(|row| time_at(row, STEP))
        .collect::<Vec<_>>();
    let (_, x, v, a) = work.model.integrate(
        execution,
        Unchecked,
        Lumped,
        &VelocityVerlet::new(Time::seconds(STEP)),
        &time,
        (deformed(&work.coordinates), work.velocities()),
        loads,
        &conditions,
    )?;
    let last = |what: &str| format!("the run on the host has {what}");
    Ok([
        flat(x.iter().last().ok_or_else(|| last("coordinates"))?),
        flat(v.iter().last().ok_or_else(|| last("velocities"))?),
        flat(a.iter().last().ok_or_else(|| last("accelerations"))?),
    ])
}

// The cost of a step on a device, and of the setup of a run, and of the forces alone, in
// milliseconds; or nothing if there is no device. The state after the first steps is checked
// against `expected`, which the host made.
fn on_a_device<B, L>(
    work: &Work<B>,
    loads: &L,
    moved: bool,
    fused: bool,
    expected: (&[Vec<f64>; 3], usize),
) -> Result<Option<(f64, f64, f64)>, AssertionError>
where
    B: DeviceBlocks + ElasticElements<3> + LumpedMassElements<3>,
    L: LoadAt<NodalCoordinates<3>, NodalForcesSolid<3>>,
{
    let (start, velocities) = (deformed(&work.coordinates), work.velocities());
    let forces = loads
        .device_forces()
        .ok_or("the loads are not ones that a device takes")?;
    let (imposed, components) = imposed_by(work, forces, moved, STEP);
    // Flattened once: doing it for every row would time the flattening, not the device.
    let started = flat(&start);
    let targets = |row| targets_at(work, &started, row, STEP);
    let schedule = schedule_of(Some(&targets), &components);
    let masses = Lumped.masses(&work.model);
    let setup_start = Instant::now();
    let Some(mut resident) = on_device(&work.model, fused) else {
        return Ok(None);
    };
    let restart = |resident: &mut Resident| {
        resident
            .start_dynamics(&masses, &start, &velocities, &imposed, &schedule)
            .map_err(describe)
    };
    restart(&mut resident)?;
    let setup = setup_start.elapsed().as_secs_f64() * 1e3;
    let (expected, steps) = expected;
    resident.steps(STEP, steps, &schedule).map_err(describe)?;
    let (x, v, a) = resident.dynamics_state().map_err(describe)?;
    close(&flat(&x), &expected[0], "coordinates")?;
    close(&flat(&v), &expected[1], "velocities")?;
    close(&flat(&a), &expected[2], "accelerations")?;
    // Enough steps to take about a second, found from a few.
    restart(&mut resident)?;
    resident.steps(STEP, 20, &schedule).map_err(describe)?;
    restart(&mut resident)?;
    let trial = Instant::now();
    resident.steps(STEP, 50, &schedule).map_err(describe)?;
    let per_step = trial.elapsed().as_secs_f64() * 1e3 / 50.0;
    let steps = ((1000.0 / per_step) as usize).clamp(50, MOST_STEPS);
    let mut step = f64::INFINITY;
    for _ in 0..repetitions() {
        restart(&mut resident)?;
        let start = Instant::now();
        resident.steps(STEP, steps, &schedule).map_err(describe)?;
        step = step.min(start.elapsed().as_secs_f64() * 1e3 / steps as f64);
    }
    resident.set_coordinates(&start).map_err(describe)?;
    let forces = time_of(repetitions(), || resident.run().map_err(describe))?;
    Ok(Some((step, setup, forces)))
}

// Times a variant of the run on a grid, and prints a row for each way of running it. A device
// that is not there ends the benchmark, which is whether to go on.
fn variant<B, L>(
    label: &str,
    work: &Work<B>,
    loads: &L,
    moved: bool,
) -> Result<bool, AssertionError>
where
    B: ParallelElasticElements<3> + LumpedMassElements<3> + DeviceBlocks,
    L: LoadAt<NodalCoordinates<3>, NodalForcesSolid<3>>,
{
    let elements = elements(&work.model);
    let rate = |step: f64| elements as f64 / step / 1e3;
    let (short, long) = history_steps(work.coordinates.len());
    let host = var("HOST").map_or(true, |host| host != "0");
    let expected = on_host(work, loads, moved, Execution::Serial, short)?;
    let (serial, threaded) = if host {
        (
            marginal(short, long, |steps| {
                black_box(on_host(work, loads, moved, Execution::Serial, steps)?);
                Ok(())
            })?,
            marginal(short, long, |steps| {
                black_box(on_host(
                    work,
                    loads,
                    moved,
                    Execution::Threads(threads()),
                    steps,
                )?);
                Ok(())
            })?,
        )
    } else {
        (f64::NAN, f64::NAN)
    };
    println!("{label}");
    println!(
        "{:>22} {:>11} {:>12} {:>10} {:>10}",
        "", "ms/step", "Melement/s", "vs threads", "setup ms"
    );
    let row = |name: &str, step: f64, setup: f64| {
        println!(
            "{name:>22} {step:>11.3} {:>12.1} {:>9.2}x {setup:>10.1}",
            rate(step),
            threaded / step
        );
    };
    if host {
        row("calling thread", serial, 0.0);
        row("threads", threaded, 0.0);
    }
    for fused in [false, true] {
        let Some((step, setup, forces)) =
            on_a_device(work, loads, moved, fused, (&expected, short))?
        else {
            return Ok(false);
        };
        let name = if fused {
            "device fused"
        } else {
            "device unfused"
        };
        row(name, step, setup);
        if fused {
            row("  forces alone", forces, 0.0);
        }
    }
    Ok(true)
}

#[test]
#[ignore]
fn steps() -> Result<(), AssertionError> {
    println!("\nthreads: {}, step {STEP:e} s", threads());
    for cells in step_cells() {
        let (model, coordinates) = grid(cells);
        let work = Work::new(model, coordinates, cells);
        println!(
            "\n{} elements, {} nodes ({cells}^3)",
            elements(&work.model),
            work.coordinates.len()
        );
        if !variant("without loads", &work, &(), false)? {
            return Ok(());
        }
        let loads = volume_loads_on(&work)?;
        if !variant("with loads", &work, &loads, false)?
            || !variant("with loads and a moved top", &work, &loads, true)?
        {
            return Ok(());
        }
    }
    Ok(())
}
