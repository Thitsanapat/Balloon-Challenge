"""Focused, simulator-free checks for experimental ZEM guidance."""

import unittest
from unittest.mock import patch

import numpy as np

from BalloonPoppingGymEnv.agents.scenario4_wind_profile_agent import (
    Scenario4WindProfileAgent,
)
from BalloonPoppingGymEnv.agents.scenario4_zem_terminal_agent import (
    Scenario4ZemTerminalAgent,
    bounded_zem_force,
    smooth_window_weight,
    zero_effort_miss_acceleration,
)
from BalloonPoppingGymEnv.evaluation.evaluate import load_scenario_parameters


class ZemGeometryTests(unittest.TestCase):
    def test_relative_zero_effort_miss(self):
        np.testing.assert_allclose(zero_effort_miss_acceleration(
            [10., 0., 0.], [-2., 0., 0.], 5.), [0., 0., 0.])
        np.testing.assert_allclose(zero_effort_miss_acceleration(
            [10., 0., 0.], [0., 0., 0.], 2.), [5., 0., 0.])
        with self.assertRaises(ValueError):
            zero_effort_miss_acceleration([1., 0., 0.], [0., 0., 0.], 0.)

    def test_time_gate_is_zero_at_boundaries_and_smooth_inside(self):
        weights = [smooth_window_weight(t, 2., 5., .4, .6)
                   for t in (1.99, 2., 2.001, 3.5, 4.999, 5., 5.01)]
        self.assertEqual(weights[0], 0.)
        self.assertEqual(weights[1], 0.)
        self.assertLess(weights[2], 1e-4)
        self.assertAlmostEqual(weights[3], .6)
        self.assertLess(weights[4], 1e-4)
        self.assertEqual(weights[5], 0.)
        self.assertEqual(weights[6], 0.)

    def test_force_blend_retains_vertical_support_and_obeys_public_envelope(self):
        force, saturated = bounded_zem_force(
            np.array([0., 0., 10.]), [100., 0., 0.], [0., 0., 0.],
            [0., 0., 0.], 2.0, 1.0, 5.0, 12.0, np.radians(75.0))
        np.testing.assert_allclose(force, [5., 0., 10.], atol=1e-9)
        self.assertFalse(saturated)
        self.assertLessEqual(np.linalg.norm(force), 12.0)
        self.assertAlmostEqual(force[2], 10.0)
        constrained, saturated = bounded_zem_force(
            np.array([0., 0., 10.]), [100., 0., 0.], [0., 0., 0.],
            [0., 0., 0.], 2.0, 1.0, 5.0, 10.0, np.radians(75.0))
        self.assertTrue(saturated)
        self.assertLessEqual(np.linalg.norm(constrained), 10.0 + 1e-9)
        self.assertGreaterEqual(constrained[2], 0.0)


class ZemAgentTests(unittest.TestCase):
    def make_agent(self):
        _, given = load_scenario_parameters(4)
        agent = Scenario4ZemTerminalAgent(
            given, launch_candidates=0, timing_candidates=0,
            search_depth=1)
        elevation = agent.elevation
        agent.target_index = 0
        agent.route = [0]
        agent.deadlines = [3.5]
        agent.plan_start = 0.0
        agent.plan_duration = 3.5
        agent.filtered_position = np.array([0., 0., elevation])
        agent.filtered_velocity = np.zeros(3)
        agent.disturbance = np.zeros(3)
        agent._zem_snapshot = (
            0.0, np.array([1]),
            np.array([[20., 0., elevation, 0., 0., 0.]]),
        )
        return agent

    def test_single_actuator_update_and_bounded_target_correction(self):
        agent = self.make_agent()
        original_jerk = np.array([1., 2., 3.])
        with patch.object(Scenario4WindProfileAgent, "_attitude_action",
                          return_value=(np.zeros(2), 0., 1.)) as parent:
            result = agent._attitude_action(
                np.array([0., 0., 1.]), original_jerk,
                10.0, np.zeros(3), 12.0)
        self.assertEqual(parent.call_count, 1)
        self.assertEqual(result[2], 1.)
        axis, jerk, magnitude = parent.call_args.args[:3]
        self.assertGreater(axis[0], 0.)
        np.testing.assert_allclose(jerk, np.zeros(3))
        self.assertLessEqual(magnitude, 12.)
        self.assertGreater(axis[2], np.cos(agent.max_tilt) - 1e-12)
        self.assertEqual(agent.diagnostics["zem_applied_steps"], 1)

    def test_unreleased_target_keeps_parent_command_unchanged(self):
        agent = self.make_agent()
        now, status, states = agent._zem_snapshot
        status[0] = 0
        axis = np.array([0., 0., 1.])
        jerk = np.array([1., 2., 3.])
        with patch.object(Scenario4WindProfileAgent, "_attitude_action",
                          return_value=(np.zeros(2), 0., 1.)) as parent:
            agent._attitude_action(axis, jerk, 10., np.zeros(3), 12.)
        self.assertEqual(parent.call_count, 1)
        np.testing.assert_array_equal(parent.call_args.args[0], axis)
        np.testing.assert_array_equal(parent.call_args.args[1], jerk)
        self.assertEqual(agent.diagnostics["zem_applied_steps"], 0)

    def test_observation_snapshot_is_cleared_even_if_parent_fails(self):
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
        self.assertIsNone(agent._zem_snapshot)


if __name__ == "__main__":
    unittest.main()
