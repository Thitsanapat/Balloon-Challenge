"""Focused geometry and actuator checks for forecast-innovation guidance."""

import unittest
from unittest.mock import patch

import numpy as np

from BalloonPoppingGymEnv.agents.scenario4_forecast_innovation_agent import (
    Scenario4ForecastInnovationAgent,
    bounded_innovation_force,
    observed_endpoint_innovation,
    smooth_innovation_gate,
)
from BalloonPoppingGymEnv.agents.scenario4_wind_profile_agent import (
    Scenario4WindProfileAgent,
)
from BalloonPoppingGymEnv.evaluation.evaluate import load_scenario_parameters


class ForecastInnovationGeometryTests(unittest.TestCase):
    def test_endpoint_agreement_produces_no_correction(self):
        class FlatWind:
            def correction(self, *_):
                return np.zeros(2)

        states = np.array([[10., 20., 100., 2., -1., 3.]])
        nominal = np.array([16., 17., 109.])
        innovation = observed_endpoint_innovation(
            states, np.array([1]), 0, 3.0, nominal, FlatWind(), 0.75)
        np.testing.assert_array_equal(innovation, np.zeros(2))
        force, saturated = bounded_innovation_force(
            [0., 0., 10.], innovation, 3.0, 0.65, 6.0, 4.0,
            1.0, 12.0, np.radians(75.0))
        np.testing.assert_array_equal(force, [0., 0., 10.])
        self.assertFalse(saturated)

    def test_released_observation_and_profile_shift_forecast(self):
        class ShearWind:
            def correction(self, *_):
                return np.array([2., -2.])

        states = np.array([[10., 20., 100., 2., -1., 3.]])
        innovation = observed_endpoint_innovation(
            states, np.array([1]), 0, 3.0,
            np.array([16., 17., 109.]), ShearWind(), 0.75)
        np.testing.assert_allclose(innovation, [1.5, -1.5])
        np.testing.assert_array_equal(observed_endpoint_innovation(
            states, np.array([0]), 0, 3.0,
            np.array([16., 17., 109.]), ShearWind(), 0.75),
            np.zeros(2))

    def test_force_direction_is_bounded_and_vertical_support_retained(self):
        force, saturated = bounded_innovation_force(
            [0., 0., 10.], [100., 0.], 1.0, 1.0,
            6.0, 4.0, 1.0, 12.0, np.radians(75.0))
        np.testing.assert_allclose(force, [4., 0., 10.])
        self.assertFalse(saturated)
        force, saturated = bounded_innovation_force(
            [0., 0., 10.], [100., 0.], 1.0, 1.0,
            6.0, 4.0, 1.0, 10.0, np.radians(75.0))
        self.assertTrue(saturated)
        self.assertAlmostEqual(force[2], 10.0)
        self.assertLessEqual(np.linalg.norm(force), 10.0 + 1e-9)

    def test_gate_full_in_practical_window_and_zero_at_edges(self):
        times = (0.19, 0.2, 0.8, 3.0, 4.4, 5.0, 5.01)
        weights = [smooth_innovation_gate(t, 0.2, 5.0, 0.6, 0.65)
                   for t in times]
        self.assertEqual(weights[0], 0.0)
        self.assertEqual(weights[1], 0.0)
        for weight in weights[2:5]:
            self.assertAlmostEqual(weight, 0.65)
        self.assertEqual(weights[5], 0.0)
        self.assertEqual(weights[6], 0.0)


class ForecastInnovationAgentTests(unittest.TestCase):
    def make_agent(self):
        _, given = load_scenario_parameters(4)
        agent = Scenario4ForecastInnovationAgent(
            given, launch_candidates=0, timing_candidates=0,
            search_depth=1)
        elevation = agent.elevation
        agent.target_index = 0
        agent.route = [0]
        agent.deadlines = [3.0]
        agent.plan_start = 0.0
        agent.plan_duration = 3.0
        agent.plan = np.zeros((6, 3))
        agent.plan[0] = [16., 0., elevation]
        agent.filtered_position = np.array([0., 0., elevation])
        agent._innovation_snapshot = (
            0.0, np.array([1]),
            np.array([[20., 0., elevation, 0., 0., 0.]]),
        )
        return agent

    def test_parent_actuators_called_once_with_corrected_axis(self):
        agent = self.make_agent()
        with patch.object(Scenario4WindProfileAgent, "_attitude_action",
                          return_value=(np.zeros(2), 0., 1.)) as parent:
            agent._attitude_action(
                np.array([0., 0., 1.]), np.ones(3),
                10.0, np.zeros(3), 12.0)
        self.assertEqual(parent.call_count, 1)
        axis, jerk, magnitude = parent.call_args.args[:3]
        self.assertGreater(axis[0], 0.0)
        np.testing.assert_array_equal(jerk, np.ones(3))
        self.assertLessEqual(magnitude, 12.0)
        self.assertEqual(agent.diagnostics["innovation_applied_steps"], 1)

    def test_no_endpoint_innovation_preserves_parent_command(self):
        agent = self.make_agent()
        # Observed target has zero drift, and its current x is the endpoint.
        agent.plan[0, 0] = 20.0
        axis = np.array([0., 0., 1.])
        jerk = np.ones(3)
        with patch.object(Scenario4WindProfileAgent, "_attitude_action",
                          return_value=(np.zeros(2), 0., 1.)) as parent:
            agent._attitude_action(axis, jerk, 10.0, np.zeros(3), 12.0)
        self.assertEqual(parent.call_count, 1)
        np.testing.assert_array_equal(parent.call_args.args[0], axis)
        np.testing.assert_array_equal(parent.call_args.args[1], jerk)
        self.assertEqual(agent.diagnostics["innovation_applied_steps"], 0)

    def test_exact_endpoint_agreement_matches_baseline_actuation(self):
        agent = self.make_agent()
        agent.plan[0, 0] = 20.0
        _, given = load_scenario_parameters(4)
        baseline = Scenario4WindProfileAgent(
            given, launch_candidates=0, timing_candidates=0,
            search_depth=1)
        axis = np.array([0., 0., 1.])
        jerk = np.array([0.1, -0.2, 0.3])
        gyro = np.array([0.01, 0.02, -0.01])
        expected = baseline._attitude_action(axis, jerk, 10., gyro, 12.)
        actual = agent._attitude_action(axis, jerk, 10., gyro, 12.)
        for observed, reference in zip(actual, expected):
            np.testing.assert_array_equal(observed, reference)
        self.assertEqual(agent.diagnostics["innovation_applied_steps"], 0)

    def test_snapshot_cleared_after_parent_failure(self):
        agent = self.make_agent()
        observation = {
            "simulation_time": 0.,
            "balloon_status": np.array([1]),
            "balloon_states": np.array([[20., 0., agent.elevation,
                                         0., 0., 0.]]),
        }
        with patch.object(Scenario4WindProfileAgent, "get_action",
                          side_effect=RuntimeError("test failure")):
            with self.assertRaisesRegex(RuntimeError, "test failure"):
                agent.get_action(observation)
        self.assertIsNone(agent._innovation_snapshot)


if __name__ == "__main__":
    unittest.main()
