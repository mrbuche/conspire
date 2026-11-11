#![cfg(all(feature = "gpu", feature = "fem"))]

//! The nodal forces of a model on the calling thread, on threads, and on a CUDA device; and, in
//! `steps`, the steps of explicit dynamics.
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
//!
//! `steps` times whole steps of velocity Verlet on a grid of hexahedra, without loads, with loads,
//! and with loads and a moved top: on the calling thread and on threads, with the integrator of
//! the library, and on a device with the state kept there. The cost of a step on the host is the
//! difference between a long run and a short one over the difference in their steps, which leaves
//! out what a run does once and is how the integrator that keeps a history is timed; on a device
//! it is the time of a long run of steps, from when the run is started, so the setup, which is
//! large, is left out and reported. `CELLS` for `steps` is `10,22,46,90` by default, and
//! `HOST=0` leaves out the host, which is slow on the large grids.

mod common;

use common::{
    Work, agree, close, components, deformed, density, disjoint, flat, imposed_by, lattice,
    loads_on, on_device, rubber, schedule_of, skip, slab, steel, targets_at,
};
use conspire::{
    constitutive::solid::hyperelastic::{NeoHookean, SaintVenantKirchhoff},
    fem::{
        Blocks, Model, NodalCoordinates, NodalReferenceCoordinates,
        block::Block,
        boundary::{BoundaryConditions, ConstrainAt, Displacement, LoadAt, Solid},
        element::linear::{Hexahedron, LinearElement},
        solid::{
            ElasticExplicitDynamics, Execution, Lumped, MassChoice, NodalForcesSolid, Unchecked,
            elastic::{ElasticElements, ParallelElasticElements, Workers},
            mass::LumpedMassElements,
        },
    },
    gpu::{Device, DeviceBlocks, GpuError},
    math::{Quantity, Tensor, integrate::VelocityVerlet},
    units::{Density, Length, Time},
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
        let mut reported = Vec::new();
        for (name, resources) in resident.resources().expect("the resources of the kernels") {
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
        resident.set_coordinates(current).expect("the coordinates");
        let kernels = time_of(repetitions, || resident.run().expect("the forces"));
        let transfers = time_of(repetitions, || {
            black_box(resident.nodal_forces(model, current).expect("the forces"));
        });
        // The call that puts the forces into nodes that are kept, which makes no nodes for them,
        // and the parts of the two calls, each timed alone: the coordinates, which are copied up
        // from the memory of the nodes; the kernels; the check for a failure; and the forces,
        // which are copied down into nodes that are made for them and into nodes that are kept.
        let mut kept = NodalForcesSolid::<3>::zero(current.len());
        let into = time_of(repetitions, || {
            resident
                .nodal_forces_into(model, current, &mut kept)
                .expect("the forces");
            black_box(&kept);
        });
        let setting = time_of(repetitions, || {
            resident.set_coordinates(current).expect("the coordinates");
        });
        let failing = time_of(repetitions, || {
            black_box(resident.failure().expect("the failures"));
        });
        let downloading = time_of(repetitions, || {
            black_box(resident.download().expect("the forces"));
        });
        let downloading_into = time_of(repetitions, || {
            resident.download_into(&mut kept).expect("the forces");
            black_box(&kept);
        });
        println!(
            "{:>30}  a call of {transfers:.3} ms, or {into:.3} ms into nodes that are kept: setting the coordinates {setting:.3}, the kernels {kernels:.3}, the failures {failing:.3}, downloading {downloading:.3}, or {downloading_into:.3} into nodes that are kept",
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

/// The step of the runs of dynamics, which is exact in binary and small enough that the grids
/// stay stable and the elements stay valid however many steps a device takes.
const STEP: f64 = 1.0 / 134_217_728.0;

/// The most steps that a device takes in a timed run, so that the motion of the top and the
/// loads of time stay small next to the cells of the smallest grids.
const MOST_STEPS: usize = 100_000;

fn step_cells() -> Vec<usize> {
    if std::env::var_os("CELLS").is_some() {
        cells()
    } else {
        vec![10, 22, 46, 90]
    }
}

/// The steps of the runs on the host that keep a history, which is large, so fewer for larger
/// grids.
fn history_steps(nodes: usize) -> (usize, usize) {
    let short = (400_000 / nodes).clamp(1, 20);
    (short, 3 * short + 1)
}

/// The cost of a step in milliseconds, from runs of `short` and `long` steps.
fn marginal(short: usize, long: usize, mut run: impl FnMut(usize)) -> f64 {
    let short_time = time_of_run(&mut run, short);
    let long_time = time_of_run(&mut run, long);
    (long_time - short_time) / (long - short) as f64
}

fn time_of_run(run: &mut impl FnMut(usize), steps: usize) -> f64 {
    time_of(repetitions(), || run(steps))
}

/// The coordinates, velocities and accelerations at the end of `steps` steps with the integrator
/// of the library, each as `3 * node + component`.
fn on_host<B, L>(
    work: &Work<B>,
    loads: &L,
    moved: bool,
    execution: Execution,
    steps: usize,
) -> [Vec<f64>; 3]
where
    B: ParallelElasticElements<3> + LumpedMassElements<3>,
    L: LoadAt<NodalCoordinates<3>, NodalForcesSolid<3>>,
{
    let base = 0..work.layer;
    let top = if moved { work.top.clone() } else { Vec::new() };
    let conditions = BoundaryConditions::<Solid<3>>::new()
        .fix_nodes(Displacement::<0>, base.clone())
        .fix_nodes(Displacement::<1>, base.clone())
        .fix_nodes(Displacement::<2>, base)
        .prescribe_nodes_with(Displacement::<0>, top, |time| {
            Length::meters(common::RATE * time.value())
        });
    let time = (0..=steps).map(time_of_step).collect::<Vec<_>>();
    let (_, x, v, a) = work
        .model
        .integrate(
            execution,
            Unchecked,
            Lumped,
            &VelocityVerlet::new(Time::seconds(STEP)),
            &time,
            (deformed(&work.coordinates), work.velocities()),
            loads,
            &conditions,
        )
        .expect("the run on the host");
    [
        flat(x.iter().last().expect("coordinates")),
        flat(v.iter().last().expect("velocities")),
        flat(a.iter().last().expect("accelerations")),
    ]
}

fn time_of_step(row: usize) -> Quantity<Time> {
    common::time_of(row, STEP)
}

/// The cost of a step on a device, and of the setup of a run, and of the forces alone, in
/// milliseconds; or nothing if there is no device. The state after the first steps is checked
/// against `expected`, which the host made.
#[allow(clippy::too_many_arguments)]
fn on_a_device<B, L>(
    work: &Work<B>,
    loads: &L,
    moved: bool,
    fused: bool,
    expected: Option<(&[Vec<f64>; 3], usize)>,
) -> Option<(f64, f64, f64)>
where
    B: DeviceBlocks + ElasticElements<3> + LumpedMassElements<3>,
    L: LoadAt<NodalCoordinates<3>, NodalForcesSolid<3>>,
{
    let (start, velocities) = (deformed(&work.coordinates), work.velocities());
    let forces = loads.device_forces().expect("loads that a device takes");
    let (imposed, components) = imposed_by(work, forces, moved, STEP);
    // Flattened once: doing it for every row would time the flattening, not the device.
    let started = flat(&start);
    let targets = |row| targets_at(work, &started, row, STEP);
    let schedule = schedule_of(Some(&targets), &components);
    let masses = Lumped.masses(&work.model);
    let setup_start = Instant::now();
    let mut resident = on_device(&work.model, fused)?;
    let restart = |resident: &mut conspire::gpu::Resident| {
        resident
            .start_dynamics(&masses, &start, &velocities, &imposed, &schedule)
            .expect("the start");
    };
    restart(&mut resident);
    let setup = setup_start.elapsed().as_secs_f64() * 1e3;
    if let Some((expected, steps)) = expected {
        resident.steps(STEP, steps, &schedule).expect("the steps");
        let (x, v, a) = resident.dynamics_state().expect("the state");
        close(&flat(&x), &expected[0], "coordinates");
        close(&flat(&v), &expected[1], "velocities");
        close(&flat(&a), &expected[2], "accelerations");
        restart(&mut resident);
    }
    // Enough steps to take about a second, found from a few.
    resident.steps(STEP, 20, &schedule).expect("the steps");
    restart(&mut resident);
    let trial = Instant::now();
    resident.steps(STEP, 50, &schedule).expect("the steps");
    let per_step = trial.elapsed().as_secs_f64() * 1e3 / 50.0;
    let steps = ((1000.0 / per_step) as usize).clamp(50, MOST_STEPS);
    let mut step = f64::INFINITY;
    for _ in 0..repetitions() {
        restart(&mut resident);
        let start = Instant::now();
        resident.steps(STEP, steps, &schedule).expect("the steps");
        step = step.min(start.elapsed().as_secs_f64() * 1e3 / steps as f64);
    }
    resident.set_coordinates(&start).expect("the coordinates");
    let forces = time_of(repetitions(), || resident.run().expect("the forces"));
    Some((step, setup, forces))
}

/// Times a variant of the run on a grid, and prints a row for each way of running it. A device
/// that is not there ends the benchmark, which is whether to go on.
fn variant<B, L>(label: &str, work: &Work<B>, loads: &L, moved: bool) -> bool
where
    B: ParallelElasticElements<3> + LumpedMassElements<3> + DeviceBlocks,
    L: LoadAt<NodalCoordinates<3>, NodalForcesSolid<3>>,
{
    let elements = elements(&work.model);
    let rate = |step: f64| elements as f64 / step / 1e3;
    let (short, long) = history_steps(work.coordinates.len());
    let host = std::env::var("HOST").map_or(true, |host| host != "0");
    let expected = on_host(work, loads, moved, Execution::Serial, short);
    let (serial, threaded) = if host {
        (
            marginal(short, long, |steps| {
                black_box(on_host(work, loads, moved, Execution::Serial, steps));
            }),
            marginal(short, long, |steps| {
                black_box(on_host(
                    work,
                    loads,
                    moved,
                    Execution::Threads(threads()),
                    steps,
                ));
            }),
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
            on_a_device(work, loads, moved, fused, Some((&expected, short)))
        else {
            return false;
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
    true
}

#[test]
#[ignore]
fn steps() {
    println!("\nthreads: {}, step {STEP:e} s", threads());
    for cells in step_cells() {
        let (model, coordinates) = grid(cells);
        let reference = NodalReferenceCoordinates::from(coordinates.clone());
        let work = Work::new(model, reference, coordinates, cells);
        println!(
            "\n{} elements, {} nodes ({cells}^3)",
            elements(&work.model),
            work.coordinates.len()
        );
        if !variant("without loads", &work, &(), false) {
            return;
        }
        let loads = loads_on(&work);
        if !variant("with loads", &work, &loads, false)
            || !variant("with loads and a moved top", &work, &loads, true)
        {
            return;
        }
    }
}

/// The sizes of the transfers in megabytes, from `TRANSFER_MB`.
fn transfer_sizes() -> Vec<usize> {
    std::env::var("TRANSFER_MB")
        .ok()
        .map(|sizes| {
            sizes
                .split(',')
                .filter_map(|size| size.parse().ok())
                .collect()
        })
        .unwrap_or_else(|| vec![1, 8, 64, 256])
}

/// What the host does around a transfer of the coordinates of the 90^3 grid: making the nodes to
/// download into, and writing into nodes that were just made and into nodes that were written
/// before. `when` says whether a device is open, which is part of what the host is doing it in.
fn on_the_host(when: &str, current: &NodalCoordinates<3>) {
    let nodes = current.len();
    let making_nodes = time_of(repetitions(), || {
        let nodes: NodalCoordinates<3> = black_box(vec![[0.0_f64; 3]; nodes]).into();
        black_box(nodes);
    });
    let source = vec![[0.5_f64; 3]; nodes];
    let fresh = time_of(repetitions(), || {
        let mut values = vec![[0.0_f64; 3]; nodes];
        values
            .as_flattened_mut()
            .copy_from_slice(source.as_flattened());
        black_box(values);
    });
    let mut reused = vec![[0.0_f64; 3]; nodes];
    let again = time_of(repetitions(), || {
        reused
            .as_flattened_mut()
            .copy_from_slice(source.as_flattened());
        black_box(&reused);
    });
    let viewing = time_of(repetitions(), || {
        black_box(black_box(current).components());
    });
    println!(
        "the host, {nodes} nodes, {when}: nodes to download into {making_nodes:.2} ms, writing nodes {fresh:.2} ms into new ones and {again:.2} ms into ones written before, the components of the nodes {viewing:.4} ms"
    );
}

#[test]
#[ignore]
fn transfers() {
    let current = deformed(&lattice(90, 90));
    on_the_host("before a device is opened", &current);

    let device = match Device::new(0) {
        Ok(device) => device,
        Err(GpuError::Unavailable(message)) => {
            skip::<()>(&format!("there is no device: {message}"));
            return;
        }
        Err(error) => panic!("{error}"),
    };
    println!("\n{}, GB/s", device.name());
    println!(
        "{:>6} | {:>9} {:>9} | {:>9} {:>9} | {:>9} {:>9} | {:>9}",
        "MB", "page up", "page down", "pin up", "pin down", "via up", "via down", "lock ms"
    );
    for megabytes in transfer_sizes() {
        let count = megabytes * 1_000_000 / 8;
        let rate = |milliseconds: f64| count as f64 * 8.0 / (milliseconds * 1e-3) / 1e9;
        let buffer = device.buffer::<f64>(count).expect("the buffer");
        let mut host = vec![1.0_f64; count];
        let page_up = time_of(repetitions(), || buffer.upload(&host).expect("the upload"));
        let page_down = time_of(repetitions(), || {
            buffer.download(&mut host).expect("the download")
        });
        let start = Instant::now();
        let mut pinned = device.pinned(count, 1.0_f64).expect("the locked memory");
        let lock = start.elapsed().as_secs_f64() * 1e3;
        let pin_up = time_of(repetitions(), || {
            buffer.upload(&pinned).expect("the upload")
        });
        let pin_down = time_of(repetitions(), || {
            buffer.download(&mut pinned).expect("the download")
        });
        // From and to a vector, which is where the values are when a run has them.
        let via_up = time_of(repetitions(), || {
            pinned.copy_from_slice(&host);
            buffer.upload(&pinned).expect("the upload");
        });
        let via_down = time_of(repetitions(), || {
            buffer.download(&mut pinned).expect("the download");
            host.copy_from_slice(&pinned);
        });
        println!(
            "{megabytes:>6} | {:>9.2} {:>9.2} | {:>9.2} {:>9.2} | {:>9.2} {:>9.2} | {lock:>9.2}",
            rate(page_up),
            rate(page_down),
            rate(pin_up),
            rate(pin_down),
            rate(via_up),
            rate(via_down),
        );
    }
    on_the_host("with a device open", &current);
}
