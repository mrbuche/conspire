#![allow(dead_code)]

use conspire::{
    constitutive::solid::hyperelastic::{NeoHookean, SaintVenantKirchhoff},
    fem::{
        Model, NodalCoordinates, NodalReferenceCoordinates, NodalVelocities,
        boundary::{DeviceForce, DeviceLoads, LoadAt, Loads, Solid},
        element::{FiniteElement, linear::LinearElement},
        solid::NodalForcesSolid,
        volume::NodalVolumeElements,
    },
    gpu::{DeviceBlocks, GpuError, Imposed, Resident, Schedule},
    math::{Quantity, Tensor, TensorRank1Vec, assert::AssertionError},
    units::{Density, Stress, Time},
};
use core::{array::from_fn, fmt::Display, ops::Range};
use std::env::var_os;

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
        var_os("CONSPIRE_REQUIRE_GPU").is_none(),
        "CONSPIRE_REQUIRE_GPU is set and {why}"
    );
    eprintln!("skipped: {why}");
    None
}

pub fn describe(error: impl Display) -> AssertionError {
    error.to_string().into()
}

pub fn on_device<B: DeviceBlocks>(model: &Model<B, 3>, fused: bool) -> Option<Resident> {
    match Resident::new(0, model, fused) {
        Ok(resident) => Some(resident),
        Err(GpuError::Unavailable(message)) => skip(&format!("there is no device: {message}")),
        Err(GpuError::KernelsNotBuilt(application)) => {
            skip(&format!("the kernels of the {application} are not built"))
        }
        Err(error) => panic!("{error}"),
    }
}

pub fn components(forces: &NodalForcesSolid<3>) -> Vec<f64> {
    flat(forces)
}

pub fn loads_on<B: NodalVolumeElements>(
    model: &Model<B, 3>,
    coordinates: &[[f64; 3]],
) -> Result<impl LoadAt<NodalCoordinates<3>, NodalForcesSolid<3>> + DeviceLoads, AssertionError> {
    let reference = NodalReferenceCoordinates::from(coordinates.to_vec());
    let forces = |scale| static_forces(coordinates, scale);
    Ok(Loads::<Solid<3>>::new()
        .nodal_forces(&reference, forces(FIRST))?
        .body_force_with(&reference, &model.nodal_volumes(), |time| {
            [
                1.0e8 * (2.0e3 * time.value()).sin(),
                -4.0e7 * (1.0 + 3.0e3 * time.value()),
                2.0e7,
            ]
            .into()
        })?
        .nodal_forces(&reference, forces(LAST))?)
}

pub const FIRST: f64 = 2.0e5;
pub const LAST: f64 = -7.0e4;

pub fn static_forces(coordinates: &[[f64; 3]], scale: f64) -> NodalForcesSolid<3> {
    coordinates
        .iter()
        .map(|&[x, y, z]| [scale * (1.0 + x), -scale * y, 0.5 * scale * (z - 0.3)])
        .collect::<Vec<_>>()
        .into()
}

pub const RATE: f64 = 20.0;

pub struct Work<B> {
    pub model: Model<B, 3>,
    pub coordinates: Vec<[f64; 3]>,
    pub layer: usize,
    pub fixed: Vec<usize>,
    pub moved: Vec<usize>,
    pub top: Vec<usize>,
}

impl<B> Work<B> {
    pub fn new(model: Model<B, 3>, coordinates: Vec<[f64; 3]>, columns: usize) -> Self {
        let layer = (columns + 1) * (columns + 1);
        let top = (coordinates.len() - layer..coordinates.len()).collect::<Vec<_>>();
        Self {
            model,
            layer,
            fixed: (0..3 * layer).collect(),
            moved: top.iter().map(|node| 3 * node).collect(),
            top,
            coordinates,
        }
    }
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

pub fn time_at(row: usize, dt: f64) -> Quantity<Time> {
    Time::seconds(row as f64 * dt)
}

pub fn volume_loads_on<B: NodalVolumeElements>(
    work: &Work<B>,
) -> Result<impl LoadAt<NodalCoordinates<3>, NodalForcesSolid<3>> + DeviceLoads, AssertionError> {
    let reference = NodalReferenceCoordinates::from(work.coordinates.clone());
    let volumes = work.model.nodal_volumes();
    Ok(Loads::<Solid<3>>::new()
        .body_force(&reference, &volumes, &[2.0e7, -1.0e7, 5.0e6].into())?
        .body_force_with(&reference, &volumes, |time| {
            [
                1.0e8 * (2.0e3 * time.value()).sin(),
                -4.0e7 * (1.0 + 3.0e3 * time.value()),
                2.0e7,
            ]
            .into()
        })?
        .body_force(&reference, &volumes, &[-7.0e6, 0.0, 3.0e6].into())?)
}

pub fn device_forces_of(loads: &impl DeviceLoads) -> Vec<DeviceForce> {
    let mut forces = Vec::new();
    loads.device_forces(&mut |force| forces.push(force));
    forces
}

pub fn imposed_by<B>(work: &Work<B>, moved: bool) -> Imposed {
    Imposed {
        fixed: work.fixed.clone(),
        moved: if moved {
            work.moved.clone()
        } else {
            Vec::new()
        },
    }
}

pub fn targets_at<B>(work: &Work<B>, start: &[f64], row: usize, dt: f64) -> Vec<f64> {
    work.moved
        .iter()
        .map(|&dof| start[dof] + RATE * time_at(row, dt).value())
        .collect()
}

pub struct Grid<'a, B> {
    work: &'a Work<B>,
    start: Vec<f64>,
    dt: f64,
}

impl<B> Schedule for Grid<'_, B> {
    fn time(&self, row: usize) -> Option<Quantity<Time>> {
        Some(time_at(row, self.dt))
    }
    fn targets(&self, row: usize) -> Option<Vec<f64>> {
        Some(targets_at(self.work, &self.start, row, self.dt))
    }
}

pub fn schedule_of<'a, B>(work: &'a Work<B>, start: &NodalCoordinates<3>, dt: f64) -> Grid<'a, B> {
    Grid {
        work,
        start: flat(start),
        dt,
    }
}

pub fn close(found: &[f64], expected: &[f64], what: &str) -> Result<(), AssertionError> {
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
    assert!(largest > 0.0, "the {what} are nothing");
    if worst <= 1e-9 * largest {
        Ok(())
    } else {
        Err(format!("the {what} differ by {worst} of up to {largest}").into())
    }
}

pub fn flat<I, U>(nodes: &TensorRank1Vec<3, I, U>) -> Vec<f64> {
    nodes
        .iter()
        .flat_map(|node| node.iter().map(|component| component.value()))
        .collect()
}

pub fn bits(values: &[f64]) -> Vec<u64> {
    values.iter().map(|value| value.to_bits()).collect()
}

pub fn agree(found: &[f64], expected: &[f64], tolerance: f64) -> Result<(), AssertionError> {
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
    if worst <= tolerance * largest {
        Ok(())
    } else {
        Err(format!("the forces differ by {worst} of up to {largest}").into())
    }
}
