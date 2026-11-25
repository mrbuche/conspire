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
//! default, and `HOST=0` leaves out the host, which is slow on the large grids. Under the fused
//! row it takes the state of the run, the coordinates, the velocities and the accelerations, into
//! nodes that are kept and into nodes that are made for it, and says how many steps each is,
//! which is how many to take between observations to keep what they cost small.
//!
//! The forces of the other tests are also timed as a call that puts them into nodes that are kept,
//! next to the call that makes nodes, with the parts of both. `transfers` times the host around a
//! transfer, before and with a device open, and the copies that a device does by size, with
//! `TRANSFER_MB` the sizes, `1,8,64,256` by default.

mod common;

use common::{
    RATE, Work, agree, close, components, deformed, density, describe, disjoint, flat, imposed_by,
    lattice, on_device, rubber, schedule_of, skip, slab, steel, time_at, volume_loads_on,
};
use conspire::{
    constitutive::solid::hyperelastic::{NeoHookean, SaintVenantKirchhoff},
    fem::{
        Blocks, Model, NodalAccelerations, NodalCoordinates, NodalReferenceCoordinates,
        NodalVelocities,
        block::Block,
        boundary::{BoundaryConditions, ConstrainAt, DeviceLoads, Displacement, LoadAt, Solid},
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
    gpu::{Device, DeviceBlocks, GpuError, Resident},
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
        let mut kept = NodalForcesSolid::<3>::zero(current.len());
        let into = time_of(repetitions, || {
            resident
                .nodal_forces_into(model, current, &mut kept)
                .map_err(describe)?;
            black_box(&kept);
            Ok::<(), AssertionError>(())
        })?;
        let setting = time_of(repetitions, || {
            resident.set_coordinates(current).map_err(describe)
        })?;
        let checking = time_of(repetitions, || {
            black_box(resident.first_error().map_err(describe)?);
            Ok::<(), AssertionError>(())
        })?;
        let downloading = time_of(repetitions, || {
            black_box(resident.download().map_err(describe)?);
            Ok::<(), AssertionError>(())
        })?;
        let downloading_into = time_of(repetitions, || {
            resident.download_into(&mut kept).map_err(describe)?;
            black_box(&kept);
            Ok::<(), AssertionError>(())
        })?;
        println!(
            "{:>30}  a call of {transfers:.3} ms, or {into:.3} ms into nodes that are kept: setting the coordinates {setting:.3}, the kernels {kernels:.3}, the errors {checking:.3}, downloading {downloading:.3}, or {downloading_into:.3} into nodes that are kept",
            "",
        );
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
        let model = Model::<Shape, 3>::from((
            Shape::from(($model, density(), connectivity, &reference)),
            reference,
        ));
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

const STEP: f64 = 1.0 / 134_217_728.0;

const MOST_STEPS: usize = 100_000;

fn step_cells() -> Vec<usize> {
    if var_os("CELLS").is_some() {
        cells()
    } else {
        vec![10, 22, 46, 90]
    }
}

fn history_steps(nodes: usize) -> (usize, usize) {
    let short = (400_000 / nodes).clamp(1, 20);
    (short, 3 * short + 1)
}

fn marginal(
    short: usize,
    long: usize,
    mut run: impl FnMut(usize) -> Result<(), AssertionError>,
) -> Result<f64, AssertionError> {
    let short_time = time_of(repetitions(), || run(short))?;
    let long_time = time_of(repetitions(), || run(long))?;
    Ok((long_time - short_time) / (long - short) as f64)
}

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

struct Cost {
    step: f64,
    setup: f64,
    forces: f64,
    observing_into: f64,
    observing: f64,
}

fn on_a_device<B, L>(
    work: &Work<B>,
    loads: &L,
    moved: bool,
    fused: bool,
    expected: (&[Vec<f64>; 3], usize),
) -> Result<Option<Cost>, AssertionError>
where
    B: DeviceBlocks + ElasticElements<3> + LumpedMassElements<3>,
    L: LoadAt<NodalCoordinates<3>, NodalForcesSolid<3>> + DeviceLoads,
{
    let (start, velocities) = (deformed(&work.coordinates), work.velocities());
    let imposed = imposed_by(work, moved);
    let schedule = schedule_of(work, &start, STEP);
    let masses = Lumped.masses(&work.model);
    let setup_start = Instant::now();
    let Some(mut resident) = on_device(&work.model, fused) else {
        return Ok(None);
    };
    let restart = |resident: &mut Resident| {
        resident
            .start_dynamics(&masses, &start, &velocities, &imposed, loads, &schedule)
            .map_err(describe)
    };
    restart(&mut resident)?;
    let setup = setup_start.elapsed().as_secs_f64() * 1e3;
    let (expected, steps) = expected;
    resident
        .steps(STEP, steps, loads, &schedule)
        .map_err(describe)?;
    let (x, v, a) = resident.dynamics_state().map_err(describe)?;
    close(&flat(&x), &expected[0], "coordinates")?;
    close(&flat(&v), &expected[1], "velocities")?;
    close(&flat(&a), &expected[2], "accelerations")?;
    restart(&mut resident)?;
    resident
        .steps(STEP, 20, loads, &schedule)
        .map_err(describe)?;
    restart(&mut resident)?;
    let trial = Instant::now();
    resident
        .steps(STEP, 50, loads, &schedule)
        .map_err(describe)?;
    let per_step = trial.elapsed().as_secs_f64() * 1e3 / 50.0;
    let steps = ((1000.0 / per_step) as usize).clamp(50, MOST_STEPS);
    let mut step = f64::INFINITY;
    for _ in 0..repetitions() {
        restart(&mut resident)?;
        let start = Instant::now();
        resident
            .steps(STEP, steps, loads, &schedule)
            .map_err(describe)?;
        step = step.min(start.elapsed().as_secs_f64() * 1e3 / steps as f64);
    }
    let nodes = work.coordinates.len();
    let mut kept = (
        NodalCoordinates::<3>::zero(nodes),
        NodalVelocities::<3>::zero(nodes),
        NodalAccelerations::<3>::zero(nodes),
    );
    let observing_into = time_of(repetitions(), || {
        resident
            .dynamics_state_into(&mut kept.0, &mut kept.1, &mut kept.2)
            .map_err(describe)
    })?;
    let observing = time_of(repetitions(), || {
        black_box(resident.dynamics_state().map_err(describe)?);
        Ok::<(), AssertionError>(())
    })?;
    resident.set_coordinates(&start).map_err(describe)?;
    let forces = time_of(repetitions(), || resident.run().map_err(describe))?;
    Ok(Some(Cost {
        step,
        setup,
        forces,
        observing_into,
        observing,
    }))
}

fn variant<B, L>(
    label: &str,
    work: &Work<B>,
    loads: &L,
    moved: bool,
) -> Result<bool, AssertionError>
where
    B: ParallelElasticElements<3> + LumpedMassElements<3> + DeviceBlocks,
    L: LoadAt<NodalCoordinates<3>, NodalForcesSolid<3>> + DeviceLoads,
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
        let against = if host {
            format!("{:>9.2}x", threaded / step)
        } else {
            format!("{:>10}", "-")
        };
        println!(
            "{name:>22} {step:>11.3} {:>12.1} {against} {setup:>10.1}",
            rate(step)
        );
    };
    if host {
        row("calling thread", serial, 0.0);
        row("threads", threaded, 0.0);
    }
    for fused in [false, true] {
        let Some(cost) = on_a_device(work, loads, moved, fused, (&expected, short))? else {
            return Ok(false);
        };
        let name = if fused {
            "device fused"
        } else {
            "device unfused"
        };
        row(name, cost.step, cost.setup);
        if fused {
            row("  forces alone", cost.forces, 0.0);
            println!(
                "{:>22} the state of the run: {:.3} ms into nodes that are kept, which is {:.1} steps, and {:.3} ms into nodes that are made, which is {:.1} steps",
                "",
                cost.observing_into,
                cost.observing_into / cost.step,
                cost.observing,
                cost.observing / cost.step,
            );
        }
    }
    Ok(true)
}

fn transfer_sizes() -> Vec<usize> {
    var("TRANSFER_MB")
        .ok()
        .map(|sizes| {
            sizes
                .split(',')
                .filter_map(|size| size.parse().ok())
                .collect()
        })
        .unwrap_or_else(|| vec![1, 8, 64, 256])
}

fn on_the_host(when: &str, nodes: usize) -> Result<(), AssertionError> {
    let making = time_of(repetitions(), || {
        black_box(NodalCoordinates::<3>::zero(black_box(nodes)));
        Ok::<(), AssertionError>(())
    })?;
    let source = vec![[0.5_f64; 3]; nodes];
    let fresh = time_of(repetitions(), || {
        let mut values = vec![[0.0_f64; 3]; nodes];
        values
            .as_flattened_mut()
            .copy_from_slice(source.as_flattened());
        black_box(values);
        Ok::<(), AssertionError>(())
    })?;
    let mut reused = vec![[0.0_f64; 3]; nodes];
    let again = time_of(repetitions(), || {
        reused
            .as_flattened_mut()
            .copy_from_slice(source.as_flattened());
        black_box(&reused);
        Ok::<(), AssertionError>(())
    })?;
    println!(
        "the host, {nodes} nodes, {when}: making nodes {making:.2} ms, writing nodes {fresh:.2} ms into new ones and {again:.2} ms into ones written before"
    );
    Ok(())
}

#[test]
#[ignore]
fn transfers() -> Result<(), AssertionError> {
    let nodes = lattice(90, 90).len();
    on_the_host("before a device is opened", nodes)?;
    let device = match Device::new(0) {
        Ok(device) => device,
        Err(GpuError::Unavailable(message)) => {
            skip::<()>(&format!("there is no device: {message}"));
            return Ok(());
        }
        Err(error) => return Err(describe(error)),
    };
    println!("\n{}, GB/s", device.name());
    println!("{:>6} | {:>9} {:>9}", "MB", "up", "down");
    for megabytes in transfer_sizes() {
        let count = megabytes * 1_000_000 / 8;
        let rate = |milliseconds: f64| count as f64 * 8.0 / (milliseconds * 1e-3) / 1e9;
        let buffer = device.buffer::<f64>(count).map_err(describe)?;
        let mut host = vec![1.0_f64; count];
        let up = time_of(repetitions(), || buffer.upload(&host).map_err(describe))?;
        let down = time_of(repetitions(), || {
            buffer.download(&mut host).map_err(describe)
        })?;
        println!("{megabytes:>6} | {:>9.2} {:>9.2}", rate(up), rate(down));
    }
    on_the_host("with a device open", nodes)
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
