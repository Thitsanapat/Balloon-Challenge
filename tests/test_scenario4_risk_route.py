"""Focused tests of observation-only route-risk ranking."""

import unittest
from unittest.mock import patch

import numpy as np

from BalloonPoppingGymEnv.agents.scenario4_risk_route_agent import (
    Scenario4RiskRouteAgent,
)
from BalloonPoppingGymEnv.agents.scenario4_wind_profile_agent import (
    Scenario4WindProfileAgent,
)


def bare_agent():
    agent = Scenario4RiskRouteAgent.__new__(Scenario4RiskRouteAgent)
    agent.risk_depth_limit = 5
    agent.risk_first_horizon = 4.
    agent.risk_future_horizon = 5.
    agent.risk_first_weight = .2
    agent.risk_future_weight = .035
    agent.risk_drift_weight = .01
    agent.risk_force_soft_limit = .8
    agent.risk_force_weight = 1.5
    agent.search_depth = 14
    agent.disturbance = np.zeros(3)
    agent.diagnostics = {'risk_ranked_nodes': 0,
                         'risk_depth_capped_plans': 0}
    agent.available_acceleration = lambda now: 25.
    return agent


def node(times, acceleration=10.):
    curves = np.zeros((len(times), 6, 3))
    for i, t in enumerate(times):
        curves[i, 2, 2] = .5*acceleration*t*t
    return tuple(range(len(times))), tuple(times), curves, None, None


class RiskRouteTests(unittest.TestCase):
    def test_earlier_first_intercept_wins_equal_duration_routes(self):
        agent = bare_agent()
        agent.risk_drift_weight = agent.risk_force_weight = 0.
        states = np.zeros((2, 6))
        early = agent._node_priority(node((3., 5.)), states, [0, 1], 24.)
        late = agent._node_priority(node((6., 2.)), states, [0, 1], 24.)
        self.assertLess(early, late)

    def test_observed_drift_and_rocket_margin_affect_ranking(self):
        agent = bare_agent()
        agent.risk_first_weight = agent.risk_future_weight = 0.
        states = np.zeros((1, 6))
        candidate = node((5.,), acceleration=15.)
        base = agent._node_priority(candidate, states, [0], 24.)
        states[0, 3] = 10.
        drifting = agent._node_priority(candidate, states, [0], 24.)
        self.assertGreater(drifting, base)
        harder = agent._node_priority(node((5.,), acceleration=23.), states, [0], 24.)
        self.assertGreater(harder, drifting)

    def test_zero_weights_leave_parent_duration_priority(self):
        agent = bare_agent()
        agent.risk_first_weight = agent.risk_future_weight = 0.
        agent.risk_drift_weight = agent.risk_force_weight = 0.
        states = np.zeros((2, 6))
        self.assertEqual(agent._node_priority(node((3., 5.)), states, [0, 1], 24.), 8.)

    def test_depth_cap_is_scoped_to_one_planning_call_even_on_failure(self):
        agent = bare_agent()
        observation = {'balloon_states': np.zeros((1, 6)),
                       'balloon_status': np.ones(1, dtype=int)}
        with patch.object(Scenario4WindProfileAgent, '_make_plan',
                          side_effect=lambda *args: self.assertEqual(agent.search_depth, 5)):
            agent._make_plan(observation, np.zeros(3), np.zeros(3), 24.)
        self.assertEqual(agent.search_depth, 14)
        self.assertEqual(agent.diagnostics['risk_depth_capped_plans'], 1)
        with patch.object(Scenario4WindProfileAgent, '_make_plan',
                          side_effect=RuntimeError('test')):
            with self.assertRaises(RuntimeError):
                agent._make_plan(observation, np.zeros(3), np.zeros(3), 24.)
        self.assertEqual(agent.search_depth, 14)


if __name__ == '__main__':
    unittest.main()
