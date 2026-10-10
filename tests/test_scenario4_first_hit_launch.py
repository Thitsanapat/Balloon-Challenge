"""First-intercept launch ranking uses only planned flight and observations."""

import unittest
from unittest.mock import patch

import numpy as np

from BalloonPoppingGymEnv.agents.chain_launch_agent import ChainLaunchAgent
from BalloonPoppingGymEnv.agents.scenario4_first_hit_launch_agent import (
    Scenario4FirstHitLaunchAgent,
)
from BalloonPoppingGymEnv.agents.submission_sqp_v2 import (
    G, PhysicsGuidanceAgent,
)


def constant_acceleration_curve(duration, horizontal_acceleration=0.0):
    curve = np.zeros((6, 3))
    curve[2] = (np.array([horizontal_acceleration, 0.0, 1.0])
                * duration**2 / 2)
    return curve


def bare_agent():
    agent = Scenario4FirstHitLaunchAgent.__new__(Scenario4FirstHitLaunchAgent)
    agent.launch_margin_weight = 1.0
    agent.launch_margin_floor = 0.25
    agent.launch_beyond_profile_weight = 2.0
    agent.profile_horizon = 8.0
    agent.disturbance = np.zeros(3)
    agent.available_acceleration = lambda now: 13.0
    agent.route = [0]
    agent.deadlines = [6.0]
    agent.plan = constant_acceleration_curve(6.0)
    return agent


class FirstHitLaunchTests(unittest.TestCase):
    def test_default_launch_key_still_prefers_longer_route(self):
        agent = ChainLaunchAgent.__new__(ChainLaunchAgent)
        long_route = agent._launch_candidate_key(None, 9, 30.0, {}, 42.0)
        short_route = agent._launch_candidate_key(None, 7, 20.0, {}, 42.0)
        self.assertLess(long_route, short_route)

    def test_low_late_lateral_margin_costs_more(self):
        agent = bare_agent()
        easy = agent._launch_candidate_key(None, 8, 25.0, {}, 0.0)
        agent.plan = constant_acceleration_curve(6.0, 7.0)
        tight = agent._launch_candidate_key(None, 8, 25.0, {}, 0.0)
        self.assertLess(easy, tight)
        self.assertLess(agent._first_leg_headroom(6.0, 0.0), 0.25)

    def test_unsupported_long_forecast_is_penalized(self):
        agent = bare_agent()
        agent.deadlines = [9.0]
        agent.plan = constant_acceleration_curve(9.0)
        long_cost = agent._launch_candidate_key(None, 8, 25.0, {}, 0.0)[1]
        agent.launch_beyond_profile_weight = 0.0
        unpenalized = agent._launch_candidate_key(None, 8, 25.0, {}, 0.0)[1]
        self.assertAlmostEqual(long_cost - unpenalized, 2.0)

    def test_shorter_first_hit_can_win_with_fewer_planned_targets(self):
        agent = bare_agent()
        agent.launched = False
        agent.launch_selected = False
        agent.launch_time = 42.0
        agent.launch_candidates = 1
        agent.elevation = 20.0
        agent.acceleration = np.zeros(3)
        agent.launch_comparisons = []
        vertical = np.array([0.0, 0.0, 1.0])
        tilted = np.array([0.2, 0.0, np.sqrt(0.96)])
        observation = {
            "simulation_time": 42.0,
            "balloon_states": np.zeros((9, 6)),
            "balloon_status": np.ones(9, dtype=int),
        }

        def fake_plan(obs, position, velocity, now):
            tilted_candidate = agent.acceleration[0] > 0.1
            count, duration = (7, 5.0) if tilted_candidate else (9, 9.0)
            agent.route = list(range(count))
            agent.deadlines = [now + duration]
            agent.plan = constant_acceleration_curve(duration)

        with patch("BalloonPoppingGymEnv.agents.chain_launch_agent.launch_axes",
                   return_value=[vertical, tilted]), patch.object(
                       Scenario4FirstHitLaunchAgent, "_make_plan",
                       side_effect=fake_plan), patch.object(
                       PhysicsGuidanceAgent, "get_action", return_value={}):
            ChainLaunchAgent.get_action(agent, observation)
        self.assertTrue(agent.launch_selected)
        self.assertEqual(len(agent.launch_comparisons), 2)
        self.assertLess(agent.launch_attitude[0], 90.0)

    def test_no_planned_route_is_ranked_last(self):
        agent = bare_agent()
        agent.route = []
        agent.plan = None
        self.assertEqual(agent._launch_candidate_key(None, 0, float("inf"), {}, 0.0)[0], 1)


if __name__ == "__main__":
    unittest.main()
