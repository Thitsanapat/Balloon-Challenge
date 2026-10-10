"""Geometry and compatibility checks for observation-only corridor ranking."""

import unittest

import numpy as np

from BalloonPoppingGymEnv.agents.scenario4_swept_corridor_agent import (
    Scenario4SweptCorridorAgent,
    swept_opportunity,
)
from BalloonPoppingGymEnv.evaluation.evaluate import load_scenario_parameters


def line(start, end):
    curve = np.zeros((6, 3))
    curve[0] = start
    curve[1] = np.asarray(end) - start
    return curve


class SweptCorridorTests(unittest.TestCase):
    def test_same_place_at_different_time_does_not_count(self):
        path = line(np.zeros(3), np.array([10., 0., 0.]))[None]
        aligned = np.array([[5., 0., 0., 0., 0., 0.]])
        moving_away = np.array([[5., 0., 0., 10., 0., 0.]])
        hit = swept_opportunity(path, [1.], aligned, [0], [], 1.5)
        miss = swept_opportunity(path, [1.], moving_away, [0], [], 1.5)
        self.assertGreater(hit, 0.8)
        self.assertLess(miss, 0.001)

    def test_each_balloon_counted_once_and_route_excluded(self):
        curves = np.stack((line(np.zeros(3), np.array([10., 0., 0.])),
                           line(np.array([10., 0., 0.]), np.zeros(3))))
        states = np.array([[5., 0., 0., 0., 0., 0.]])
        score = swept_opportunity(curves, [1., 1.], states, [0], [], 1.5)
        self.assertGreater(score, 0.8)
        self.assertLessEqual(score, 1.0)
        self.assertEqual(swept_opportunity(
            curves, [1., 1.], states, [0], [0], 1.5), 0.0)
        self.assertEqual(swept_opportunity(
            curves, [1., 1.], states, [], [], 1.5), 0.0)

    def test_zero_weight_preserves_original_beam_order(self):
        _, given = load_scenario_parameters(4)
        agent = Scenario4SweptCorridorAgent(
            given, corridor_weight=0., launch_candidates=0,
            timing_candidates=0)
        curve = line(np.zeros(3), np.array([10., 0., 0.]))
        node = ((0,), (2.,), curve[None], np.zeros(3), np.zeros(3))
        self.assertEqual(agent._node_priority(
            node, np.zeros((2, 6)), np.array([0, 1]), 42.), 2.)
        self.assertEqual(agent.diagnostics["corridor_ranked_nodes"], 0)

    def test_conservative_gate_needs_group_and_caps_bonus(self):
        _, given = load_scenario_parameters(4)
        agent = Scenario4SweptCorridorAgent(
            given, corridor_weight=10., corridor_min_opportunity=1.,
            corridor_max_bonus=0.5, corridor_legs=1,
            launch_candidates=0, timing_candidates=0)
        curve = line(np.zeros(3), np.array([10., 0., 0.]))
        node = ((0,), (1.,), curve[None], np.zeros(3), np.zeros(3))
        states = np.array([[0., 0., 0., 0., 0., 0.],
                           [3., 0., 0., 0., 0., 0.],
                           [6., 0., 0., 0., 0., 0.]])
        self.assertEqual(agent._node_priority(
            node, states, np.array([0, 1]), 42.), 1.)
        self.assertAlmostEqual(agent._node_priority(
            node, states, np.array([0, 1, 2]), 42.), 0.5)

    def test_invalid_settings_rejected(self):
        _, given = load_scenario_parameters(4)
        with self.assertRaises(ValueError):
            Scenario4SweptCorridorAgent(given, corridor_width=0.)


if __name__ == "__main__":
    unittest.main()
