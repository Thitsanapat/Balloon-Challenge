"""The escape must require a wide observed gap and a usable new route."""

import unittest
from unittest.mock import patch

import numpy as np

from BalloonPoppingGymEnv.agents.chain_beam_agent import ChainBeamAgent
from BalloonPoppingGymEnv.agents.scenario4_hopeless_leg_agent import (
    Scenario4HopelessLegAgent,
    optimistic_horizontal_miss_lower_bound,
)


def active_agent():
    agent = Scenario4HopelessLegAgent.__new__(Scenario4HopelessLegAgent)
    agent.route = [0, 1]
    agent.deadlines = [1.0, 4.0]
    agent.ends_v = [np.zeros(3), np.zeros(3)]
    agent.ends_a = [np.zeros(3), np.zeros(3)]
    agent.plan = np.arange(18.0).reshape(6, 3)
    agent.plan_start = 0.0
    agent.plan_duration = 1.0
    agent.target_index = 0
    agent.failed_until = {}
    agent._escape_retry_after = {}
    agent.escape_max_remaining = 4.0
    agent.escape_clearance = 2.0
    agent.escape_rocket_acceleration_slack = 3.0
    agent.escape_target_acceleration_slack = 4.0
    agent.escape_cooldown = 1.5
    agent.escape_retry_interval = 0.8
    agent.escape_depth = 3
    agent.escape_beam_width = 8
    agent.escape_branch_targets = 8
    agent.search_depth = 14
    agent.beam_width = 12
    agent.branch_targets = 8
    agent.radius = 1.5
    agent.max_tilt = np.radians(75.0)
    agent.launch_time = 0.0
    agent.burn_time = 20.0
    agent.thrust = 20.0
    agent.initial_mass = 1.0
    agent.dry_mass = 1.0
    agent.mass_flow = 0.0
    agent.diagnostics = dict(escape_refresh_failures=0,
                             escape_out_of_reach=0,
                             escape_alternative_attempts=0,
                             escape_alternative_accepted=0)
    return agent


def observation(distance=40.0):
    return {
        "balloon_states": np.array([
            [distance, 0.0, 0.0, 0.0, 0.0, 0.0],
            [6.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]),
        "balloon_status": np.array([1, 1]),
    }


class HopelessLegTests(unittest.TestCase):
    def test_reach_bound_allows_early_pass_and_generous_acceleration(self):
        early_pass = optimistic_horizontal_miss_lower_bound(
            [10.0, 0.0], [-20.0, 0.0], 1.0, 0.0)
        self.assertLess(early_pass, 1e-9)
        strict = optimistic_horizontal_miss_lower_bound(
            [30.0, 0.0], [0.0, 0.0], 1.0, 0.0)
        generous = optimistic_horizontal_miss_lower_bound(
            [30.0, 0.0], [0.0, 0.0], 1.0, 20.0)
        self.assertAlmostEqual(strict, 30.0)
        self.assertLessEqual(generous, 20.0)
        self.assertGreater(generous, 19.0)
        self.assertGreater(strict, generous)

    def test_success_commits_alternative_and_restores_search_limits(self):
        agent = active_agent()
        observed = observation()
        saved_states = observed["balloon_states"].copy()

        def choose_alternative(instance, obs, position, velocity, now):
            self.assertEqual(instance.route, [])
            self.assertGreater(instance.failed_until[0], now)
            self.assertEqual(instance.search_depth, 3)
            instance.route = [1]
            instance.deadlines = [2.0]
            instance.ends_v = [np.zeros(3)]
            instance.ends_a = [np.zeros(3)]
            instance.plan = np.zeros((6, 3))
            instance.target_index = 1

        with patch.object(ChainBeamAgent, "_make_plan", choose_alternative):
            accepted = agent._rescue_committed_leg(
                observed, np.zeros(3), np.zeros(3), 0.0, 1.0)
        self.assertTrue(accepted)
        self.assertEqual(agent.route, [1])
        self.assertEqual(agent.target_index, 1)
        self.assertEqual(agent.search_depth, 14)
        self.assertEqual(agent.beam_width, 12)
        self.assertEqual(agent.diagnostics["escape_alternative_accepted"], 1)
        np.testing.assert_array_equal(observed["balloon_states"], saved_states)

    def test_failed_alternative_keeps_old_plan_and_route(self):
        agent = active_agent()
        original_plan = agent.plan.copy()
        with patch.object(ChainBeamAgent, "_make_plan", return_value=None):
            accepted = agent._rescue_committed_leg(
                observation(), np.zeros(3), np.zeros(3), 0.0, 1.0)
        self.assertFalse(accepted)
        self.assertEqual(agent.route, [0, 1])
        self.assertEqual(agent.target_index, 0)
        self.assertEqual(agent.failed_until, {})
        self.assertEqual(agent.search_depth, 14)
        self.assertEqual(agent.diagnostics["escape_alternative_attempts"], 1)
        np.testing.assert_array_equal(agent.plan, original_plan)

    def test_reachable_target_does_not_start_alternative_search(self):
        agent = active_agent()
        with patch.object(ChainBeamAgent, "_make_plan") as search:
            accepted = agent._rescue_committed_leg(
                observation(distance=10.0), np.zeros(3), np.zeros(3),
                0.0, 1.0)
        self.assertFalse(accepted)
        search.assert_not_called()
        self.assertEqual(agent.route, [0, 1])
        self.assertEqual(agent.diagnostics["escape_alternative_attempts"], 0)


if __name__ == "__main__":
    unittest.main()
