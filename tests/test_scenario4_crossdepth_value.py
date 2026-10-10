import unittest
from unittest.mock import patch

import numpy as np

from BalloonPoppingGymEnv.agents.scenario4_crossdepth_value_agent import (
    Scenario4CrossDepthValueAgent, discounted_capture_value,
)
from BalloonPoppingGymEnv.agents.submission_scenario4_pure_joint_v2 import (
    Scenario4PureJointRefreshAgent,
)


class CrossDepthValueTests(unittest.TestCase):
    def test_short_reliable_route_can_beat_long_fragile_one(self):
        short = discounted_capture_value([.9, .9], [2., 2.], 20., .08)
        long = discounted_capture_value([.2]*4, [3.]*4, 20., .08)
        self.assertGreater(short, long)

    def test_miss_is_not_assumed_to_terminate_every_subsequent_capture(self):
        self.assertGreater(discounted_capture_value([0., 1.], [1., 1.], 20., 0.), 0.)

    def test_invalid_values_rejected(self):
        for probabilities, durations in [([1.1], [1.]), ([.5], [-1.]),
                                         ([float('nan')], [1.])]:
            with self.assertRaises(ValueError):
                discounted_capture_value(probabilities, durations, 20., .08)

    def make_agent(self):
        a = Scenario4CrossDepthValueAgent.__new__(Scenario4CrossDepthValueAgent)
        a.launched = True
        a.route_events, a.value_events = [], []
        a.value_adoption_margin, a.proposals_per_depth = .05, 3
        a.terminal_weight, a.acceleration = 0., np.zeros(3)
        a.diagnostics = dict(value_nodes_scored=0, value_pool_peak=0,
                             value_routes_adopted=0, value_shorter_adopted=0)
        a._route_value = lambda node, states, now: .9 if node[0] == (7,) else .5
        a._select = lambda *args: None
        return a

    def test_actual_planner_boundary_selects_shorter_densely_valid_proposal(self):
        a = self.make_agent()
        short = ((7,), (2.,), np.zeros((1, 6, 3)), None, None)
        long = ((1, 2), (2., 2.), np.zeros((2, 6, 3)), None, None)

        def parent(_, observation, position, velocity, now):
            a._node_priority(short, np.zeros((8, 6)), np.arange(8), now)
            a._node_priority(long, np.zeros((8, 6)), np.arange(8), now)
            a.route, a.deadlines = [1, 2], [now+2., now+4.]
            a.route_events.append([now, [1, 2], 4.])

        a._valid = lambda *args, **kwargs: (
            np.array([True]), None, np.zeros((1, 3)), np.zeros((1, 3)))
        with patch.object(Scenario4PureJointRefreshAgent, '_make_plan', parent), \
                patch('BalloonPoppingGymEnv.agents.scenario4_crossdepth_value_agent.free_chain',
                      return_value=long[2]):
            a._make_plan({}, np.zeros(3), np.zeros(3), 50.)
        self.assertEqual(a.route, [7])
        self.assertEqual(a.diagnostics['value_shorter_adopted'], 1)
        self.assertIsNone(a._value_pool)

    def test_high_value_cannot_bypass_dense_physics_gate(self):
        a = self.make_agent()
        short = ((7,), (2.,), np.zeros((1, 6, 3)), None, None)

        def parent(_, observation, position, velocity, now):
            a._node_priority(short, np.zeros((8, 6)), np.arange(8), now)
            a.route, a.deadlines = [1, 2], [now+2., now+4.]
            a.route_events.append([now, [1, 2], 4.])

        a._valid = lambda *args, **kwargs: (
            np.array([False]), None, np.zeros((1, 3)), np.zeros((1, 3)))
        with patch.object(Scenario4PureJointRefreshAgent, '_make_plan', parent), \
                patch('BalloonPoppingGymEnv.agents.scenario4_crossdepth_value_agent.free_chain',
                      return_value=np.zeros((2, 6, 3))):
            a._make_plan({}, np.zeros(3), np.zeros(3), 50.)
        self.assertEqual(a.route, [1, 2])
        self.assertEqual(a.diagnostics['value_routes_adopted'], 0)

    def test_launch_axis_planning_is_unchanged(self):
        a = self.make_agent()
        a.launched = False
        with patch.object(Scenario4PureJointRefreshAgent, '_make_plan') as parent:
            a._make_plan({}, np.zeros(3), np.zeros(3), 42.)
        parent.assert_called_once()


if __name__ == '__main__':
    unittest.main()
