#![allow(dead_code)]

use conspire::{
    constitutive::solid::hyperelastic::{NeoHookean, SaintVenantKirchhoff},
    fem::{
        Model, NodalCoordinates,
        element::{FiniteElement, linear::LinearElement},
        solid::NodalForcesSolid,
    },
    gpu::{DeviceBlocks, GpuError, Resident},
    math::{Quantity, Tensor, TensorRank1Vec},
    units::{Density, Stress},
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
    nodes
        .iter()
        .flat_map(|node| node.iter().map(|component| component.value()))
        .collect()
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
