"""Geometry, saturation and neutral-command tests for terminal correction."""

import unittest
from unittest.mock import patch

import numpy as np

from BalloonPoppingGymEnv.agents.scenario4_terminal_residual_agent import (
    Scenario4TerminalResidualAgent,
    critically_damped_tracking_residual,
    feasible_terminal_force,
    terminal_gate,
    terminal_miss_vector,
)
from BalloonPoppingGymEnv.agents.scenario4_wind_profile_agent import (
    Scenario4WindProfileAgent,
)
from BalloonPoppingGymEnv.evaluation.evaluate import load_scenario_parameters


class TerminalResidualGeometryTests(unittest.TestCase):
    def test_critical_decay_includes_current_velocity(self):
        p = np.array([2., 0., -1.])
        v = np.array([-1., 0., 0.5])
        np.testing.assert_array_equal(
            critically_damped_tracking_residual(p, v, 1.3, 0), p)
        expected = ((1 + 1.3*2)*p + 2*v)*np.exp(-1.3*2)
        np.testing.assert_allclose(
            critically_damped_tracking_residual(p, v, 1.3, 2), expected)

    def test_target_and_tracking_residual_are_both_three_dimensional(self):
        class FlatWind:
            def correction(self, *_):
                return np.zeros(2)

        states = np.array([[10., 20., 100., 2., -1., 3.]])
        miss = terminal_miss_vector(
            states, [1], 0, 2, [14., 18., 105.], [0., 0., 1.],
            FlatWind(), 0.75)
        np.testing.assert_array_equal(miss, [0., 0., 0.])
        miss = terminal_miss_vector(
            states, [1], 0, 2, [14., 18., 105.], [0., 0., -2.],
            FlatWind(), 0.75)
        np.testing.assert_array_equal(miss, [0., 0., 3.])

    def test_force_preserves_baseline_when_correction_is_unneeded_or_infeasible(self):
        args = dict(horizon=2., weight=.5, deadband=.65,
                    max_error=3., max_delta=1.5, max_vertical_delta=.8,
                    time_floor=1., available=12., max_tilt=np.radians(75.))
        base = np.array([0., 0., 10.])
        force, limited = feasible_terminal_force(base, [0., 0., 0.5], **args)
        np.testing.assert_array_equal(force, base)
        self.assertFalse(limited)
        force, limited = feasible_terminal_force(base, [2., 0., 2.], **args)
        self.assertGreater(force[0], 0)
        self.assertGreater(force[2], 10.)
        self.assertLessEqual(np.linalg.norm(force-base), 1.5 + 1e-8)
        self.assertLessEqual(force[2]-base[2], .8 + 1e-8)
        self.assertFalse(limited)
        full = np.array([0., 0., 12.])
        force, limited = feasible_terminal_force(full, [0., 0., 3.], **args)
        np.testing.assert_array_equal(force, full)
        self.assertTrue(limited)

    def test_gate_zero_outside_window(self):
        self.assertEqual(terminal_gate(0.3, .3, 3.5, .55, .35), 0.)
        self.assertEqual(terminal_gate(3.5, .3, 3.5, .55, .35), 0.)
        self.assertAlmostEqual(terminal_gate(2., .3, 3.5, .55, .35), .35)


class TerminalResidualAgentTests(unittest.TestCase):
    def make_agent(self):
        _, given = load_scenario_parameters(4)
        agent = Scenario4TerminalResidualAgent(
            given, launch_candidates=0, timing_candidates=0, search_depth=1)
        agent.target_index = 0
        agent.plan_start = 0.
        agent.plan_duration = 2.
        agent.plan = np.zeros((6, 3))
        agent.plan[0] = [10., 0., agent.elevation]
        agent.filtered_position = agent.plan[0].copy()
        agent.filtered_velocity = np.zeros(3)
        agent._terminal_snapshot = (0., np.array([1]),
            np.array([[10., 0., agent.elevation, 0., 0., 0.]]))
        return agent

    def test_nominal_agreement_is_exact_parent_command(self):
        agent = self.make_agent()
        axis = np.array([0., 0., 1.])
        jerk = np.array([.1, -.2, .3])
        with patch.object(Scenario4WindProfileAgent, '_attitude_action',
                          return_value=(np.zeros(2), 0., 1.)) as parent:
            agent._attitude_action(axis, jerk, 10., np.zeros(3), 12.)
        np.testing.assert_array_equal(parent.call_args.args[0], axis)
        np.testing.assert_array_equal(parent.call_args.args[1], jerk)
        self.assertEqual(agent.diagnostics['terminal_applied_steps'], 0)

    def test_vertical_miss_requests_more_vertical_force(self):
        agent = self.make_agent()
        agent._terminal_snapshot = (0., np.array([1]),
            np.array([[10., 0., agent.elevation + 3., 0., 0., 0.]]))
        with patch.object(Scenario4WindProfileAgent, '_attitude_action',
                          return_value=(np.zeros(2), 0., 1.)) as parent:
            agent._attitude_action(np.array([0., 0., 1.]), np.ones(3),
                                   10., np.zeros(3), 12.)
        self.assertGreater(parent.call_args.args[2], 10.)
        np.testing.assert_array_equal(parent.call_args.args[1], np.ones(3))
        self.assertEqual(agent.diagnostics['terminal_applied_steps'], 1)

    def test_snapshot_cleared_on_parent_error(self):
        agent = self.make_agent()
        observation = dict(simulation_time=0., balloon_status=np.array([1]),
            balloon_states=np.array([[10., 0., agent.elevation, 0., 0., 0.]]))
        with patch.object(Scenario4WindProfileAgent, 'get_action',
                          side_effect=RuntimeError('failure')):
            with self.assertRaisesRegex(RuntimeError, 'failure'):
                agent.get_action(observation)
        self.assertIsNone(agent._terminal_snapshot)


if __name__ == '__main__':
    unittest.main()
