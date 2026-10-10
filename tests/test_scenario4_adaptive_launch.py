"""Tests for seed-independent, observation-triggered launch readiness."""

import unittest
from unittest.mock import patch

import numpy as np

from BalloonPoppingGymEnv.agents.scenario4_adaptive_launch_agent import (
    Scenario4AdaptiveLaunchAgent,
    observed_launch_group,
)
from BalloonPoppingGymEnv.agents.scenario4_wind_profile_agent import (
    Scenario4WindProfileAgent,
)
from BalloonPoppingGymEnv.evaluation.evaluate import load_scenario_parameters


class AdaptiveLaunchTests(unittest.TestCase):
    def setUp(self):
        _, self.given = load_scenario_parameters(4)
        self.states = np.array([
            [0., 0., 55., 0., 0., 0.],
            [2., 0., 55., 0., 0., 0.],
            [0., 2., 55., 0., 0., 0.],
            [2., 2., 55., 0., 0., 0.],
        ])

    def test_reachable_observed_group_and_time_alignment(self):
        count, duration = observed_launch_group(
            self.states, np.arange(4), [0., 0., 20.], 12.0,
            group_radius=4.)
        self.assertEqual(count, 4)
        self.assertLessEqual(duration, 7.5)
        # At the same future time, one fast-drifting balloon has left the group.
        moving = self.states.copy()
        moving[3, 3] = 10.0
        count, _ = observed_launch_group(
            moving, np.arange(4), [0., 0., 20.], 12.0,
            group_radius=4.)
        self.assertEqual(count, 3)

    def test_early_trigger_uses_observation_not_seed(self):
        agent = Scenario4AdaptiveLaunchAgent(
            self.given, launch_group_radius=4., launch_candidates=0,
            timing_candidates=0)
        observation = {"simulation_time": 40.0,
                       "balloon_states": self.states,
                       "balloon_status": np.ones(4, dtype=int)}
        with patch.object(Scenario4WindProfileAgent, "get_action",
                          return_value={"launch": False}):
            agent.get_action(observation)
        self.assertEqual(agent.launch_time, 40.0)
        self.assertEqual(agent.diagnostics["adaptive_launch_reason"],
                         "observed_group")
        self.assertEqual(agent.diagnostics["adaptive_launch_group"], 4)

    def test_default_when_no_group(self):
        agent = Scenario4AdaptiveLaunchAgent(
            self.given, launch_candidates=0, timing_candidates=0)
        observation = {"simulation_time": 42.0,
                       "balloon_states": self.states,
                       "balloon_status": np.zeros(4, dtype=int)}
        with patch.object(Scenario4WindProfileAgent, "get_action",
                          return_value={"launch": False}):
            agent.get_action(observation)
        self.assertEqual(agent.launch_time, 42.0)
        self.assertEqual(agent.diagnostics["adaptive_launch_reason"],
                         "default_time")


if __name__ == "__main__":
    unittest.main()
