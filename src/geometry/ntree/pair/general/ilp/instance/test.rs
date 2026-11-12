use super::Instance;

#[test]
fn isolated_cell_costs_one() {
    let instance = Instance::new(vec![([0, 0], true)], Default::default());
    let (assignment, cost) = instance.solve_bruteforce();
    assert_eq!(cost, 1);
    assert!(instance.feasible(&assignment));
}

#[test]
fn two_adjacent_cells_cost_two() {
    let instance = Instance::new(vec![([0, 0], true), ([1, 0], true)], Default::default());
    let (assignment, cost) = instance.solve_bruteforce();
    assert_eq!(cost, 2);
    assert!(instance.feasible(&assignment));
}

#[test]
fn three_in_a_row_cannot_avoid_extra_cost() {
    let instance = Instance::new(
        vec![([0, 0], true), ([1, 0], true), ([2, 0], true)],
        Default::default(),
    );
    let (assignment, cost) = instance.solve_bruteforce();
    assert_eq!(cost, 3);
    assert!(instance.feasible(&assignment));
}

#[test]
fn unrequired_neighbor_is_never_assigned() {
    let instance = Instance::new(vec![([0, 0], true), ([1, 0], false)], Default::default());
    let (assignment, cost) = instance.solve_bruteforce();
    assert_eq!(cost, 1);
    assert!(instance.feasible(&assignment));
}

#[test]
fn solve_matches_bruteforce_on_examples() {
    for cells in [
        vec![([0, 0], true)],
        vec![([0, 0], true), ([1, 0], true)],
        vec![([0, 0], true), ([1, 0], true), ([2, 0], true)],
        vec![([0, 0], true), ([1, 0], false)],
    ] {
        let instance = Instance::new(cells, Default::default());
        let (assignment, cost) = instance.solve();
        let (_, bruteforce_cost) = instance.solve_bruteforce();
        assert!(instance.feasible(&assignment));
        assert_eq!(cost, bruteforce_cost);
    }
}

fn fuzz_cells_2d(seed: u64) -> Vec<([i32; 2], bool)> {
    let mut state = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    let mut rand = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (state >> 33) as usize
    };
    (0..3)
        .flat_map(|x| (0..2).map(move |y| [x, y]))
        .map(|cell| (cell, rand() % 2 == 0))
        .collect()
}

#[test]
fn solve_matches_bruteforce_fuzz() {
    for seed in 0..200u64 {
        let cells = fuzz_cells_2d(seed);
        if !cells.iter().any(|(_, required)| *required) {
            continue;
        }
        let instance = Instance::new(cells, Default::default());
        let (assignment, cost) = instance.solve();
        let (_, bruteforce_cost) = instance.solve_bruteforce();
        assert!(
            instance.feasible(&assignment),
            "seed {seed}: solve produced infeasible assignment"
        );
        assert_eq!(
            cost, bruteforce_cost,
            "seed {seed}: solve cost {cost} != bruteforce cost {bruteforce_cost}"
        );
    }
}

#[test]
fn refused_vertices_are_never_assigned() {
    for seed in 0..200u64 {
        let cells = fuzz_cells_2d(seed);
        if !cells.iter().any(|(_, required)| *required) {
            continue;
        }
        let refused: std::collections::HashSet<[i32; 2]> = cells
            .iter()
            .flat_map(|&(cell, _)| Instance::<2>::vertices_of(cell))
            .filter(|vertex| vertex.iter().any(|coordinate| coordinate % 2 == 0))
            .filter(|vertex| (seed as i32 + 31 * vertex[0] + 17 * vertex[1]) % 2 == 0)
            .collect();
        let instance = Instance::new(cells, refused.clone());
        let (assignment, _) = instance.solve();
        assert!(instance.feasible(&assignment), "seed {seed}");
        assert!(
            assignment.iter().all(|vertex| !refused.contains(vertex)),
            "seed {seed}: assigned a refused vertex"
        );
    }
}

fn fuzz_cells<const D: usize>(seed: u64, extent: [i32; D]) -> Vec<([i32; D], bool)> {
    let mut state = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    let mut rand = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (state >> 33) as usize
    };
    let count: i32 = extent.iter().product();
    (0..count)
        .map(|code| {
            let mut rest = code;
            let cell = core::array::from_fn(|axis| {
                let coordinate = rest % extent[axis];
                rest /= extent[axis];
                coordinate
            });
            (cell, rand() % 2 == 0)
        })
        .collect()
}

fn matches_bruteforce<const D: usize>(extent: [i32; D], seeds: u64) {
    for seed in 0..seeds {
        let cells = fuzz_cells(seed, extent);
        if !cells.iter().any(|(_, required)| *required) {
            continue;
        }
        let instance = Instance::new(cells, Default::default());
        let (assignment, cost) = instance.solve();
        let (_, bruteforce_cost) = instance.solve_bruteforce();
        assert!(instance.feasible(&assignment), "seed {seed}");
        assert_eq!(cost, bruteforce_cost, "seed {seed}");
    }
}

#[test]
fn solve_matches_bruteforce_on_larger_grids() {
    matches_bruteforce([4, 3], 100);
}

#[test]
fn solve_matches_bruteforce_in_three_dimensions() {
    matches_bruteforce([2, 2, 1], 100);
}
