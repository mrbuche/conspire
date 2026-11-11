#![allow(dead_code)]

use conspire::{
    constitutive::solid::hyperelastic::{NeoHookean, SaintVenantKirchhoff},
    fem::{
        Model, NodalCoordinates, NodalReferenceCoordinates, NodalVelocities,
        boundary::{DeviceForce, LoadAt, Loads, Solid},
        element::{FiniteElement, linear::LinearElement},
        solid::NodalForcesSolid,
        volume::NodalVolumeElements,
    },
    gpu::{DeviceBlocks, GpuError, Imposed, ImposedLoad, Resident, Schedule},
    math::{Quantity, Tensor, TensorRank1Vec},
    units::{Density, Stress, Time},
};
use core::{array::from_fn, ops::Range};

pub fn lattice(columns: usize, layers: usize) -> Vec<[f64; 3]> {
    (0..=layers)
        .flat_map(|k| {
            (0..=columns).flat_map(move |j| {
                (0..=columns).map(move |i| {
                    [
                        i as f64 / columns as f64,
                        j as f64 / columns as f64,
                        k as f64 / columns as f64,
                    ]
                })
            })
        })
        .collect()
}

pub fn slab(columns: usize, layers: Range<usize>) -> Vec<[usize; 8]> {
    let index = |i: usize, j: usize, k: usize| (k * (columns + 1) + j) * (columns + 1) + i;
    layers
        .flat_map(|k| {
            (0..columns).flat_map(move |j| {
                (0..columns).map(move |i| {
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

pub fn density() -> Quantity<Density> {
    Density::kilograms_per_cubic_meter(7.8e3)
}

pub fn rubber() -> NeoHookean {
    NeoHookean {
        shear_modulus: Stress::pascals(3.0e9),
        bulk_modulus: Stress::pascals(13.0e9),
    }
}

pub fn steel() -> SaintVenantKirchhoff {
    SaintVenantKirchhoff {
        shear_modulus: Stress::pascals(80.0e9),
        bulk_modulus: Stress::pascals(160.0e9),
    }
}

pub fn deformed(coordinates: &[[f64; 3]]) -> NodalCoordinates<3> {
    coordinates
        .iter()
        .map(|&[x, y, z]| {
            [
                x + 3e-2 * (3.0 * y).sin(),
                y + 2e-2 * (2.0 * z).sin(),
                z + 4e-2 * (4.0 * x).sin(),
            ]
        })
        .collect::<Vec<_>>()
        .into()
}

pub fn disjoint<const G: usize, const N: usize>(count: usize) -> (Vec<[usize; N]>, Vec<[f64; 3]>)
where
    LinearElement<G, N>: FiniteElement<G, 3, N, N>,
{
    let parametric = <LinearElement<G, N> as FiniteElement<G, 3, N, N>>::parametric_reference();
    let (mut connectivity, mut coordinates) = (Vec::new(), Vec::new());
    for element in 0..count {
        let mut nodes = [0; N];
        for (a, xi) in parametric.iter().enumerate() {
            let xi: [f64; 3] = from_fn(|k| xi[k].value());
            nodes[a] = coordinates.len();
            let wobble = |i: usize| 0.03 * (1.7 * a as f64 + i as f64 + element as f64).sin();
            coordinates.push([
                1.1 * xi[0] + 0.2 * xi[1] - 0.1 * xi[2] + wobble(0),
                0.05 * xi[0] + 0.9 * xi[1] + 0.15 * xi[2] + wobble(1),
                -0.1 * xi[0] + 0.12 * xi[1] + 1.2 * xi[2] + wobble(2),
            ]);
        }
        connectivity.push(nodes);
    }
    (connectivity, coordinates)
}

pub fn skip<T>(why: &str) -> Option<T> {
    assert!(
        std::env::var_os("CONSPIRE_REQUIRE_GPU").is_none(),
        "CONSPIRE_REQUIRE_GPU is set and {why}"
    );
    eprintln!("skipped: {why}");
    None
}

pub fn on_device<B: DeviceBlocks>(model: &Model<B, 3>, fused: bool) -> Option<Resident> {
    match Resident::new(0, model, fused) {
        Ok(resident) => Some(resident),
        Err(GpuError::Unavailable(message)) => skip(&format!("there is no device: {message}")),
        Err(GpuError::KernelsNotBuilt) => skip("the kernels are not built"),
        Err(error) => panic!("{error}"),
    }
}

pub fn components(forces: &NodalForcesSolid<3>) -> Vec<f64> {
    forces
        .iter()
        .flat_map(|force| force.iter().map(|component| component.value()))
        .collect()
}

pub fn flat<I, U>(nodes: &TensorRank1Vec<3, I, U>) -> Vec<f64> {
    conspire::gpu::flatten(nodes)
}

pub fn bits(values: &[f64]) -> Vec<u64> {
    values.iter().map(|value| value.to_bits()).collect()
}

pub fn agree(found: &[f64], expected: &[f64], tolerance: f64) {
    assert_eq!(found.len(), expected.len());
    let largest = expected
        .iter()
        .fold(0.0_f64, |largest, x| largest.max(x.abs()));
    let worst = found
        .iter()
        .zip(expected)
        .fold(0.0_f64, |worst, (found, expected)| {
            worst.max((found - expected).abs())
        });
    assert!(largest > 1e3, "the forces are {largest}, which is nothing");
    assert!(
        worst <= tolerance * largest,
        "the forces differ by {worst} of up to {largest}"
    );
}

/// A step that a time grid of these has is exact in binary, so that the steps of the integrator
/// of the host, which takes each from the difference of times, are all this.
pub const DT: f64 = 1.0 / 1_048_576.0;

/// The rate at which the top of a body moves, in meters per second.
pub const RATE: f64 = 20.0;

/// A body on a grid of hexahedra, held at its bottom layer and moved at its top.
pub struct Work<B> {
    pub model: Model<B, 3>,
    pub reference: NodalReferenceCoordinates<3>,
    pub coordinates: Vec<[f64; 3]>,
    /// The nodes in a layer.
    pub layer: usize,
    /// The degrees of freedom of the nodes of the bottom layer.
    pub fixed: Vec<usize>,
    /// The `x` coordinates of the nodes of the top layer.
    pub moved: Vec<usize>,
    /// The nodes of the top layer.
    pub top: Vec<usize>,
}

impl<B> Work<B> {
    /// A body of `model` on a lattice of `columns` by `columns` columns of nodes, in layers.
    pub fn new(
        model: Model<B, 3>,
        reference: NodalReferenceCoordinates<3>,
        coordinates: Vec<[f64; 3]>,
        columns: usize,
    ) -> Self {
        let layer = (columns + 1) * (columns + 1);
        let top = (coordinates.len() - layer..coordinates.len()).collect::<Vec<_>>();
        Self {
            model,
            reference,
            layer,
            fixed: (0..3 * layer).collect(),
            moved: top.iter().map(|node| 3 * node).collect(),
            top,
            coordinates,
        }
    }
    /// A velocity that varies from node to node, and is nothing where the body is fixed.
    pub fn velocities(&self) -> NodalVelocities<3> {
        self.coordinates
            .iter()
            .enumerate()
            .map(|(node, &[x, y, z])| {
                if node < self.layer {
                    [0.0; 3]
                } else {
                    [
                        0.7 * (2.0 * y + z).sin(),
                        0.4 * (3.0 * z).cos(),
                        0.9 * x * y,
                    ]
                }
            })
            .collect::<Vec<_>>()
            .into()
    }
}

/// The time of the row `row` of a grid of times of the step `dt`.
pub fn time_of(row: usize, dt: f64) -> Quantity<Time> {
    Time::seconds(row as f64 * dt)
}

/// The loads on a body: a force on the volume that does not change, a force on the volume that
/// is a function of time, and another that does not change, in that order, which is the order
/// they are subtracted in. They are forces on the volume so that the accelerations they make do
/// not depend on how fine the mesh is.
pub fn loads_on<B: NodalVolumeElements>(
    work: &Work<B>,
) -> impl LoadAt<NodalCoordinates<3>, NodalForcesSolid<3>> {
    let volumes = work.model.nodal_volumes();
    Loads::<Solid<3>>::new()
        .body_force(&work.reference, &volumes, &[2.0e7, -1.0e7, 5.0e6].into())
        .expect("the first load")
        .body_force_with(&work.reference, &volumes, |time| {
            [
                1.0e8 * (2.0e3 * time.value()).sin(),
                -4.0e7 * (1.0 + 3.0e3 * time.value()),
                2.0e7,
            ]
            .into()
        })
        .expect("the load of time")
        .body_force(&work.reference, &volumes, &[-7.0e6, 0.0, 3.0e6].into())
        .expect("the last load")
}

/// The components of the loads of time, as a function of the row of the grid of times.
pub type Components<'a> = Vec<Box<dyn Fn(usize) -> [f64; 3] + 'a>>;

/// What is imposed on a run, from the forces that a device takes, and the components of the loads
/// of time on a grid of times of the step `dt`; and the coordinates that are moved if `moved`.
pub fn imposed_by<'a, B>(
    work: &Work<B>,
    forces: Vec<DeviceForce<'a>>,
    moved: bool,
    dt: f64,
) -> (Imposed, Components<'a>) {
    let mut loads = Vec::new();
    let mut components: Components = Vec::new();
    for force in forces {
        match force {
            DeviceForce::Constant(values) => loads.push(ImposedLoad::Constant(values)),
            DeviceForce::Weighted {
                weights,
                components: of_time,
            } => {
                loads.push(ImposedLoad::Weighted(weights));
                components.push(Box::new(move |row| of_time(time_of(row, dt))));
            }
        }
    }
    let moved = if moved {
        work.moved.clone()
    } else {
        Vec::new()
    };
    let imposed = Imposed {
        fixed: work.fixed.clone(),
        moved,
        loads,
    };
    (imposed, components)
}

/// Where the coordinates that are moved are held at a row: where they start and the rate times
/// the time, which is what the host holds a prescribed displacement at.
pub fn targets_at<B>(work: &Work<B>, start: &[f64], row: usize, dt: f64) -> Vec<f64> {
    work.moved
        .iter()
        .map(|&dof| start[dof] + RATE * time_of(row, dt).value())
        .collect()
}

pub fn schedule_of<'a>(
    targets: Option<&'a dyn Fn(usize) -> Vec<f64>>,
    components: &'a Components<'_>,
) -> Schedule<'a> {
    Schedule {
        targets,
        components: components
            .iter()
            .map(|component| &**component as &dyn Fn(usize) -> [f64; 3])
            .collect(),
    }
}

/// That the values agree to a fraction of the largest of them.
pub fn close(found: &[f64], expected: &[f64], what: &str) {
    let largest = expected
        .iter()
        .fold(0.0_f64, |largest, x| largest.max(x.abs()));
    let worst = found
        .iter()
        .zip(expected)
        .fold(0.0_f64, |worst, (found, expected)| {
            worst.max((found - expected).abs())
        });
    assert!(largest > 0.0, "the {what} are nothing");
    assert!(
        worst <= 1e-9 * largest,
        "the {what} differ by {worst} of up to {largest}"
    );
}
