use super::{Conservative, ConservativeAt, Load, LoadAt, Loads};
use crate::{
    domain::boundary::{Field, Solid},
    math::Quantity,
    units::Time,
};

struct Pull(f64);

impl LoadAt<[f64; 2], [f64; 2]> for Pull {
    fn subtract_at(
        &self,
        _: Quantity<Time>,
        state: &[f64; 2],
        residual: &mut [f64; 2],
    ) -> Result<(), String> {
        self.subtract_from(state, residual)
    }
}

impl Load<[f64; 2], [f64; 2]> for Pull {
    fn subtract_from(&self, _: &[f64; 2], residual: &mut [f64; 2]) -> Result<(), String> {
        residual[0] -= self.0;
        Ok(())
    }
}

impl ConservativeAt<[f64; 2], f64> for Pull {
    fn potential_energy_at(&self, _: Quantity<Time>, state: &[f64; 2]) -> Result<f64, String> {
        self.potential_energy(state)
    }
}

impl Conservative<[f64; 2], f64> for Pull {
    fn potential_energy(&self, state: &[f64; 2]) -> Result<f64, String> {
        Ok(-self.0 * state[0])
    }
}

struct Ramp(f64);

impl LoadAt<[f64; 2], [f64; 2]> for Ramp {
    fn subtract_at(
        &self,
        time: Quantity<Time>,
        _: &[f64; 2],
        residual: &mut [f64; 2],
    ) -> Result<(), String> {
        residual[1] -= self.0 * time.value();
        Ok(())
    }
}

struct Failing;

impl LoadAt<[f64; 2], [f64; 2]> for Failing {
    fn subtract_at(
        &self,
        _: Quantity<Time>,
        state: &[f64; 2],
        residual: &mut [f64; 2],
    ) -> Result<(), String> {
        self.subtract_from(state, residual)
    }
}

impl Load<[f64; 2], [f64; 2]> for Failing {
    fn subtract_from(&self, _: &[f64; 2], _: &mut [f64; 2]) -> Result<(), String> {
        Err("failed".to_string())
    }
}

#[test]
fn no_loads_leave_the_residual_alone() {
    let mut residual = [1.0, 2.0];
    ().subtract_from(&[0.0; 2], &mut residual).unwrap();
    assert_eq!(residual, [1.0, 2.0]);
    assert_eq!(Conservative::potential_energy(&(), &[3.0; 2]), Ok(0.0));
}

#[test]
fn loads_add_up() {
    let loads = Loads::<Solid<3>>::new().with(Pull(2.0)).with(Pull(3.0));
    let mut residual = [10.0, 0.0];
    loads.0.subtract_from(&[0.0; 2], &mut residual).unwrap();
    assert_eq!(residual, [5.0, 0.0]);
    assert_eq!(loads.0.potential_energy(&[4.0, 0.0]), Ok(-20.0));
}

#[test]
fn a_failing_load_fails_the_nest() {
    let loads = Loads::<Field<()>>::new().with(Pull(1.0)).with(Failing);
    assert_eq!(
        loads.0.subtract_from(&[0.0; 2], &mut [0.0; 2]),
        Err("failed".to_string())
    );
}

#[test]
fn loads_that_depend_on_time_take_the_time_they_are_given() {
    let loads = Loads::<Solid<3>>::new().with(Pull(2.0)).with(Ramp(3.0));
    let mut residual = [10.0, 10.0];
    loads
        .0
        .subtract_at(Time::seconds(4.0), &[0.0; 2], &mut residual)
        .unwrap();
    assert_eq!(residual, [8.0, -2.0]);
}

#[test]
fn static_loads_ignore_the_time() {
    let loads = Loads::<Solid<3>>::new().with(Pull(2.0));
    let (mut early, mut late) = ([10.0, 0.0], [10.0, 0.0]);
    loads
        .0
        .subtract_at(Time::seconds(0.0), &[0.0; 2], &mut early)
        .unwrap();
    loads
        .0
        .subtract_at(Time::seconds(9.0), &[0.0; 2], &mut late)
        .unwrap();
    assert_eq!(early, late);
}

#[cfg(feature = "gpu")]
mod device {
    use super::super::{DeviceForce, DeviceLoads, Loads};
    use crate::{domain::boundary::Solid, math::Quantity, units::Time};

    struct Held(Vec<f64>);

    impl DeviceLoads for Held {
        fn device_forces(&self, each: &mut impl FnMut(DeviceForce)) {
            each(DeviceForce::Constant(self.0.clone()));
        }
        fn device_weighted(&self) -> usize {
            0
        }
        fn device_components(&self, which: usize, _: Quantity<Time>) -> [f64; 3] {
            panic!("there is no force of time {which}")
        }
    }

    struct Ramped {
        weights: Vec<f64>,
        scale: f64,
    }

    impl DeviceLoads for Ramped {
        fn device_forces(&self, each: &mut impl FnMut(DeviceForce)) {
            each(DeviceForce::Weighted(self.weights.clone()));
        }
        fn device_weighted(&self) -> usize {
            1
        }
        fn device_components(&self, which: usize, time: Quantity<Time>) -> [f64; 3] {
            assert_eq!(which, 0, "there is only one force of time");
            let time = time.value();
            [
                self.scale * time,
                2.0 * self.scale * time,
                3.0 * self.scale * time,
            ]
        }
    }

    fn ramped(scale: f64) -> Ramped {
        Ramped {
            weights: vec![scale, 2.0 * scale],
            scale,
        }
    }

    fn forces_of(loads: &impl DeviceLoads) -> Vec<DeviceForce> {
        let mut forces = Vec::new();
        loads.device_forces(&mut |force| forces.push(force));
        forces
    }

    fn of_time(loads: &impl DeviceLoads, time: Quantity<Time>) -> [[f64; 3]; 2] {
        [
            loads.device_components(0, time),
            loads.device_components(1, time),
        ]
    }

    fn expected() -> Vec<DeviceForce> {
        vec![
            DeviceForce::Constant(vec![1.0]),
            DeviceForce::Weighted(vec![1.0, 2.0]),
            DeviceForce::Weighted(vec![10.0, 20.0]),
            DeviceForce::Constant(vec![2.0]),
        ]
    }

    fn nests() -> (impl DeviceLoads, impl DeviceLoads, impl DeviceLoads) {
        (
            (
                ((Held(vec![1.0]), ramped(1.0)), ramped(10.0)),
                Held(vec![2.0]),
            ),
            (
                Held(vec![1.0]),
                (ramped(1.0), (ramped(10.0), Held(vec![2.0]))),
            ),
            Loads::<Solid<3>>::new()
                .with(Held(vec![1.0]))
                .with(ramped(1.0))
                .with(ramped(10.0))
                .with(Held(vec![2.0])),
        )
    }

    #[test]
    fn nested_forces_are_given_in_the_order_that_they_are_nested_in() {
        let (left, right, built) = nests();
        assert_eq!(forces_of(&left), expected());
        assert_eq!(forces_of(&right), expected());
        assert_eq!(forces_of(&built), expected());
        assert!(forces_of(&()).is_empty());
    }

    #[test]
    fn the_forces_of_time_are_asked_for_by_their_number_among_those_of_time() {
        let (left, right, built) = nests();
        let time = Time::seconds(0.5);
        let expected = [[0.5, 1.0, 1.5], [5.0, 10.0, 15.0]];
        assert_eq!(left.device_weighted(), 2);
        assert_eq!(of_time(&left, time), expected);
        assert_eq!(right.device_weighted(), 2);
        assert_eq!(of_time(&right, time), expected);
        assert_eq!(built.device_weighted(), 2);
        assert_eq!(of_time(&built, time), expected);
        assert_eq!(().device_weighted(), 0);
    }

    #[test]
    fn the_components_of_the_rows_are_three_to_a_time_in_the_order_of_the_times() {
        let loads = (Held(vec![1.0]), (ramped(1.0), ramped(10.0)));
        let times = [Time::seconds(1.0), Time::seconds(2.0)];
        assert_eq!(
            loads.device_component_rows(0, &times),
            [1.0, 2.0, 3.0, 2.0, 4.0, 6.0]
        );
        assert_eq!(
            loads.device_component_rows(1, &times),
            [10.0, 20.0, 30.0, 20.0, 40.0, 60.0]
        );
        assert!(loads.device_component_rows(0, &[]).is_empty());
    }

    #[test]
    #[should_panic(expected = "there is only one force of time")]
    fn a_force_of_time_that_there_is_not_is_not_given() {
        (Held(vec![1.0]), ramped(1.0)).device_components(1, Time::seconds(1.0));
    }

    #[test]
    #[should_panic(expected = "there are no forces")]
    fn no_forces_have_no_force_of_time() {
        ().device_components(0, Time::seconds(1.0));
    }

    #[test]
    fn a_force_has_a_value_for_each_degree_of_freedom_or_a_weight_for_each_node() {
        let constant = |values| DeviceForce::Constant(vec![0.0; values]);
        let weighted = |weights| DeviceForce::Weighted(vec![0.0; weights]);
        assert!(constant(6).check(2).is_ok());
        assert!(weighted(2).check(2).is_ok());
        let message = constant(5).check(2).unwrap_err();
        assert!(
            message.contains("5 values") && message.contains("6 degrees"),
            "{message}"
        );
        let message = weighted(3).check(2).unwrap_err();
        assert!(
            message.contains("3 weights") && message.contains("2 nodes"),
            "{message}"
        );
    }
}
