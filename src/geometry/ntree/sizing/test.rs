use crate::{
    geometry::{
        mesh::test::sphere,
        ntree::{CurvatureSizing, Octree, Sizing},
    },
    math::Quantity,
};

#[test]
fn a_size_field_knows_the_depth_it_asks_for() {
    let tessellation = sphere(4, 8, 2.0);
    let ordinary = Sizing::new(&tessellation, 4.0, CurvatureSizing::default(), 0);
    assert!(ordinary.levels() <= 15);
    assert!(ordinary.fits::<u16>());
    assert!(Octree::<u16, usize>::refine(&ordinary).is_ok());
    let deep = Sizing::new(&tessellation, 1.0e6, CurvatureSizing::default(), 0);
    assert!(deep.levels() > 15);
    assert!(!deep.fits::<u16>());
    assert!(deep.fits::<u32>());
    assert_eq!(
        Octree::<u16, usize>::refine(&deep).err(),
        Some("sizing field exceeds maximum octree depth")
    );
}

#[test]
fn a_thickness_floor_limits_the_depth_it_asks_for() {
    let tessellation = sphere(4, 8, 2.0);
    let free = Sizing::new(&tessellation, 4.0, CurvatureSizing::default(), 0);
    let below = Sizing::new(
        &tessellation,
        4.0,
        CurvatureSizing {
            thickness_floor: Some(Quantity::new(1.0e-6)),
            ..Default::default()
        },
        0,
    );
    assert_eq!(below.levels(), free.levels());
    assert_eq!(below.targets, free.targets);
    let above = Sizing::new(
        &tessellation,
        4.0,
        CurvatureSizing {
            thickness_floor: Some(Quantity::new(16.0)),
            ..Default::default()
        },
        0,
    );
    assert!(above.levels() < free.levels());
}
