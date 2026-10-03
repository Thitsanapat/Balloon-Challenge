"""Climb ranking changes only the ordering of already-feasible beam nodes."""

import unittest

import numpy as np

from BalloonPoppingGymEnv.agents.scenario4_vertical_energy_agent import (
    Scenario4VerticalEnergyAgent,
)


def _node(duration, exit_climb, target=0):
    return ((target,), (duration,), None, np.zeros(3),
            np.array([0.0, 0.0, exit_climb]))


def _agent(weight, available=12.0, reserve=0.8):
    agent = Scenario4VerticalEnergyAgent.__new__(Scenario4VerticalEnergyAgent)
    agent.vertical_energy_weight = weight
    agent.reserve = reserve
    agent.available_acceleration = lambda now: available
    return agent


class Scenario4VerticalEnergyTests(unittest.TestCase):
    def test_zero_weight_preserves_original_order_and_skips_energy_estimate(self):
        agent = _agent(0.0)
        states = np.zeros((1, 6))
        states[0, 5] = 5.0
        self.assertEqual(agent._node_priority(_node(4.0, -10.0), states, [0], 24.0), 4.0)
        self.assertEqual(agent._node_priority(_node(4.5, 5.0), states, [0], 24.0), 4.5)

    def test_positive_weight_can_prefer_a_climbing_exit(self):
        agent = _agent(0.5)
        states = np.zeros((1, 6))
        states[0, 5] = 5.0
        early_descent = agent._node_priority(_node(4.0, -1.0), states, [0], 24.0)
        later_climb = agent._node_priority(_node(4.5, 5.0), states, [0], 24.0)
        self.assertEqual(early_descent, 6.0)  # four-second recovery cap
        self.assertEqual(later_climb, 4.5)
        self.assertLess(later_climb, early_descent)

    def test_recovery_uses_remaining_vertical_thrust_after_reserve(self):
        agent = _agent(0.5)
        states = np.zeros((1, 6))
        states[0, 5] = 5.0
        node = _node(4.0, 3.0)
        # 12 - 9.80665 - 0.8 = 1.39335 m/s^2 usable climb acceleration.
        expected = 4.0 + 0.5 * 2.0 / 1.39335
        self.assertAlmostEqual(agent._node_priority(node, states, [0], 24.0), expected)
        agent.reserve = 0.0
        self.assertLess(agent._node_priority(node, states, [0], 24.0), expected)

    def test_nonclimbing_target_does_not_demand_upward_target_velocity(self):
        agent = _agent(0.5)
        states = np.zeros((1, 6))
        states[0, 5] = -2.0
        self.assertEqual(agent._node_priority(_node(4.0, 0.0), states, [0], 24.0), 4.0)

    def test_invalid_weight_is_rejected_before_parent_initialization(self):
        for weight in (-0.1, float("nan"), float("inf")):
            with self.subTest(weight=weight):
                with self.assertRaises(ValueError):
                    Scenario4VerticalEnergyAgent(None, vertical_energy_weight=weight)


if __name__ == "__main__":
    unittest.main()
