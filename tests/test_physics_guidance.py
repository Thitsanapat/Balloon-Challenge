"""Check physical invariants independently of the simulator's scoring code."""

import numpy as np
import unittest

from BalloonPoppingGymEnv.agents.physics_guidance_agent import (
    G, PhysicsGuidanceAgent, allocate_acceleration, quintic_intercept, sample_curve,
)
from BalloonPoppingGymEnv.evaluation.evaluate import load_scenario_parameters


def test_quintic_satisfies_six_boundary_conditions():
    p0 = np.array([2., -3., 20.])
    v0 = np.array([4., 1., 7.])
    a0 = np.array([0.1, -0.2, 1.])
    target = np.array([14., 6., 30.])
    terminal_velocity = np.array([2., 0., 1.])
    for duration in [0.4, 3., 12.]:
        curve = quintic_intercept(p0, v0, a0, target, terminal_velocity, duration)
        p, v, a, _ = sample_curve(curve, duration, [0., duration])
        np.testing.assert_allclose(p, [p0, target], atol=1e-9)
        np.testing.assert_allclose(v, [v0, terminal_velocity], atol=1e-9)
        np.testing.assert_allclose(a, [a0, np.zeros(3)], atol=1e-9)


def test_vertical_support_not_lost_when_lateral_force_saturates():
    result = allocate_acceleration([50., -30., 10.], 12., np.radians(55.))
    assert result[2] == 10.
    assert np.linalg.norm(result) <= 12. + 1e-10
    assert np.isclose(np.linalg.norm(result[:2]), np.sqrt(44.))
    np.testing.assert_array_equal(allocate_acceleration([1, 2, 10], 0, 0.5), [0, 0, 0])


def test_rejects_interception_requiring_impossible_thrust():
    _, given = load_scenario_parameters(1)
    agent = PhysicsGuidanceAgent(given)
    curve = quintic_intercept([0, 0, 30], [0, 0, 0], [0, 0, 0],
                             [100, 0, 30], [0, 0, 0], 1.)
    assert not agent._feasible(curve, 1., 4.)
    hover = quintic_intercept([0, 0, 30], [0, 0, 0], [0, 0, 0],
                             [0, 0, 30], [0, 0, 0], 1.)
    assert agent._feasible(hover, 1., 4.)
    assert not agent._feasible(hover, 1., 34.)


def test_known_circle_requires_v_squared_over_radius():
    # A 10 m radius circle at 30 m/s requires 90 m/s^2 before gravity.
    _, given = load_scenario_parameters(1)
    agent = PhysicsGuidanceAgent(given)
    demand = np.linalg.norm(np.array([30**2/10, 0, 0])-G)
    assert demand > 7*agent.available_acceleration(4.)


def test_actions_finite_and_observation_not_mutated_before_launch():
    _, given = load_scenario_parameters(1)
    agent = PhysicsGuidanceAgent(given)
    obs = {"simulation_time": 0., "rocket_sensors": np.full(12, np.nan),
           "balloon_states": np.zeros((100, 6)), "balloon_status": np.zeros((100, 1), dtype=int)}
    action = agent.get_action(obs)
    assert not action["launch"]
    assert all(np.all(np.isfinite(value)) for value in action.values())
    assert np.isnan(obs["rocket_sensors"]).all()
    assert not obs["balloon_status"].any()


def test_tvc_and_throttle_rate_limits_apply_to_successive_actions():
    _, given = load_scenario_parameters(1)
    agent = PhysicsGuidanceAgent(given)
    previous_tvc = np.zeros(2)
    previous_throttle = 1.
    for axis in ([1., 0., 0.], [-1., 0., 0.], [0., 1., 0.], [0., 0., 1.]):
        tvc, roll, throttle = agent._attitude_action(np.array(axis), np.zeros(3),
                                                   5., np.zeros(3), 12.)
        assert np.all(np.abs(tvc-previous_tvc) <= agent.control["gimbal_rate_limit"]*agent.dt+1e-12)
        assert abs(throttle-previous_throttle) <= agent.control["throttle_rate_limit"]*agent.dt+1e-12
        assert np.all(np.isfinite(tvc)) and np.isfinite(roll)
        previous_tvc, previous_throttle = tvc, throttle


def load_tests(loader, tests, pattern):
    return unittest.TestSuite(
        unittest.FunctionTestCase(function)
        for name, function in globals().items()
        if name.startswith("test_") and callable(function)
    )


if __name__ == "__main__":
    unittest.main()
