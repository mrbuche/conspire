#![allow(dead_code)]

//! The meshes, models and checks that the tests of a device share.

use conspire::{
    constitutive::solid::hyperelastic::{NeoHookean, SaintVenantKirchhoff},
    fem::{
        Model, NodalCoordinates,
        element::{FiniteElement, linear::LinearElement},
        solid::NodalForcesSolid,
    },
    gpu::{DeviceBlocks, GpuError, Options, Resident},
    math::{Quantity, Tensor},
    units::{Density, Stress},
};
use core::{array::from_fn, ops::Range};

/// The nodes of a grid of `columns` by `columns` columns of hexahedra, in `layers` layers.
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

/// Hexahedra filling the layers `layers` of a grid of `columns` by `columns` columns.
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

/// A smooth, small displacement so that every element has a nonzero stress.
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

/// Elements that share no nodes, each the parametric element of its shape bent a little
/// differently, which are valid by construction.
///
/// They lie on top of each other, which does not matter as they share no nodes, and keeps the
/// coordinates the size of an element. Far from the origin, in the size of an element, the
/// deformation gradient is a sum that cancels, and the last bits of the coordinates become
/// differences between the host and a device that fuses multiplications and additions.
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

/// The model on the first device, with the operations of the kernels fused or not.
pub fn on_device<B: DeviceBlocks>(model: &Model<B, 3>, fused: bool) -> Option<Resident> {
    on_device_with(
        model,
        &Options {
            fused,
            ..Options::default()
        },
    )
}

/// The model on the first device, with the kernels compiled as the options say.
pub fn on_device_with<B: DeviceBlocks>(model: &Model<B, 3>, options: &Options) -> Option<Resident> {
    match Resident::new(0, model, options) {
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

/// That the forces agree to a fraction of the largest of them.
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
