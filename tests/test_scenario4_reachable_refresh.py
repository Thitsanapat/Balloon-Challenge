import unittest
from unittest.mock import patch

import numpy as np

from BalloonPoppingGymEnv.agents.scenario4_reachable_refresh_agent import (
    Scenario4ReachableRefreshAgent, preview_actuator_shortfall,
)
from BalloonPoppingGymEnv.agents.submission_sqp_v2 import boundary_curve


class PreviewActuatorShortfallTests(unittest.TestCase):
    def setUp(self):
        self.limit = lambda now: 30.0
        self.curve = boundary_curve(
            np.array([0.0, 0.0, 10.0]), np.zeros(3),
            np.zeros(3), np.array([12.0, 0.0, 10.0]),
            np.zeros(3), np.zeros(3), 2.0)

    def preview(self, axis, throttle=1.0, limit=None):
        return preview_actuator_shortfall(
            self.curve, 2.0, 0.0, np.zeros(3), np.asarray(axis),
            throttle, self.limit if limit is None else limit, 0.2, 0.2)

    def test_lagged_axis_increases_shortfall(self):
        aligned = self.preview([0.7, 0.0, 0.714])
        lagged = self.preview([-0.7, 0.0, 0.714])
        self.assertGreater(lagged, aligned)

    def test_reduced_throttle_increases_shortfall(self):
        self.assertGreater(self.preview([0, 0, 1], throttle=0.1),
                           self.preview([0, 0, 1], throttle=1.0))

    def test_expired_thrust_increases_shortfall(self):
        self.assertGreater(self.preview([0, 0, 1], limit=lambda now: 0.0),
                           self.preview([0, 0, 1]))

    def test_invalid_inputs_rejected(self):
        with self.assertRaises(ValueError):
            self.preview([0, 0, 0])


class CommittedRefreshTests(unittest.TestCase):
    def make_agent(self):
        agent = Scenario4ReachableRefreshAgent.__new__(
            Scenario4ReachableRefreshAgent)
        agent.route = [0, 1]
        agent.deadlines = [4.0, 6.0]
        agent.launch_time = 0.0
        agent.burn_time = 30.0
        agent.acceleration = np.zeros(3)
        agent.terminal_weight = 0.0
        agent.reach_delays = (0.0, 0.4)
        agent._reach_fallback = None
        agent.diagnostics = dict(reach_candidate_attempts=0,
                                 reach_refreshed=0,
                                 reach_infeasible_joint_refresh=0,
                                 reach_retained_nominal=0)
        return agent

    def observation(self):
        return dict(balloon_states=np.array([[0., 0., 10., 1., 0., 1.],
                                             [5., 0., 12., 1., 0., 1.]]),
                    balloon_status=np.ones(2))

    def test_infeasible_nominal_reoptimizes_full_route_without_delay(self):
        agent = self.make_agent()
        with patch.object(agent, '_valid', return_value=(
                np.ones(2, bool), None, np.zeros((2, 3)),
                np.zeros((2, 3)))), patch.object(agent, '_preview',
                return_value=0.0), patch.object(agent, '_select') as select:
            self.assertTrue(agent._rescue_committed_leg(
                self.observation(), np.zeros(3), np.zeros(3), 1.0, 3.0))
        self.assertEqual(agent.route, [0, 1])
        self.assertEqual(agent.deadlines, [4.0, 6.0])
        self.assertEqual(agent.diagnostics['reach_infeasible_joint_refresh'], 1)
        self.assertEqual(select.call_args.args[0], 0)

    def test_failed_preview_rescue_keeps_feasible_nominal_curve(self):
        agent = self.make_agent()
        curve = np.zeros((6, 3))
        agent._reach_fallback = (curve, 3.0, 1.0, 1.0)
        with patch.object(agent, '_valid', return_value=(
                np.zeros(2, bool), None, None, None)), \
                patch.object(agent, '_select') as select:
            self.assertTrue(agent._rescue_committed_leg(
                self.observation(), np.zeros(3), np.zeros(3), 1.0, 3.0))
        self.assertIs(select.call_args.args[1], curve)
        self.assertEqual(agent.diagnostics['reach_retained_nominal'], 1)
        self.assertEqual(agent.deadlines, [4.0, 6.0])

    def test_observed_shortfall_can_select_less_demanding_delayed_chain(self):
        agent = self.make_agent()
        nominal = np.zeros((6, 3))
        agent._reach_fallback = (nominal, 3.0, 1.0, 1.0)
        with patch.object(agent, '_valid', return_value=(
                np.ones(2, bool), None, np.zeros((2, 3)),
                np.zeros((2, 3)))), patch.object(agent, '_preview',
                side_effect=(0.7, 0.1)), patch.object(agent, '_select'):
            self.assertTrue(agent._rescue_committed_leg(
                self.observation(), np.zeros(3), np.zeros(3), 1.0, 3.0))
        np.testing.assert_allclose(agent.deadlines, [4.4, 6.4])
        self.assertEqual(agent.diagnostics['reach_refreshed'], 1)
        self.assertEqual(agent.diagnostics['reach_infeasible_joint_refresh'], 0)


if __name__ == '__main__':
    unittest.main()
