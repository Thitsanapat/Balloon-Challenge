import unittest
from types import SimpleNamespace

import numpy as np
import yaml

from BalloonPoppingGymEnv.agents.scenario4_headroom_learned_route_agent import (
    Scenario4HeadroomLearnedRouteAgent,
)


class HeadroomLearnedRouteTests(unittest.TestCase):
    def agent(self, threshold=0.8):
        agent = object.__new__(Scenario4HeadroomLearnedRouteAgent)
        agent.launched = True
        agent.risk_penalty_seconds = 0.5
        agent._risk_planning_state = (np.zeros(3), np.zeros(3), 42.0)
        agent.max_required_total_fraction = threshold
        agent.available_acceleration = lambda now: 10.0
        agent.capture_model = SimpleNamespace(probability=lambda features: 0.2)
        agent.diagnostics = {'risk_ranked_nodes': 0, 'risk_headroom_skips': 0}
        return agent

    def test_nearly_full_thrust_candidate_cannot_be_reranked(self):
        agent = self.agent()
        states = np.array([[10., 0., 0., 0., 0., 0.]])
        node = ((0,), (2.0,), None, None, None)
        self.assertEqual(agent._node_priority(node, states, [0], 42.), 2.0)
        self.assertEqual(agent.diagnostics['risk_headroom_skips'], 1)
        self.assertEqual(agent.diagnostics['risk_ranked_nodes'], 0)

    def test_feasible_headroom_retains_learned_penalty(self):
        agent = self.agent()
        agent.available_acceleration = lambda now: 20.0
        states = np.array([[0., 0., 5., 0., 0., 0.]])
        node = ((0,), (3.0,), None, None, None)
        self.assertAlmostEqual(agent._node_priority(node, states, [0], 42.),
                               3.0 + 0.5 * 0.8)
        self.assertEqual(agent.diagnostics['risk_ranked_nodes'], 1)
        self.assertEqual(agent.diagnostics['risk_headroom_skips'], 0)

    def test_candidate_config_keeps_selected_control_settings(self):
        from pathlib import Path
        root = Path(__file__).resolve().parents[1]
        selected = yaml.safe_load((root / 'BalloonPoppingGymEnv/evaluation/configs/submission_scenario4_guarded_innovation_launch42.yaml').read_text())
        candidate = yaml.safe_load((root / 'BalloonPoppingGymEnv/evaluation/configs/scenario4_headroom_learned_route_launch42.yaml').read_text())
        for key, value in selected['agent_kwargs'].items():
            self.assertEqual(candidate['agent_kwargs'][key], value, key)
        self.assertEqual(candidate['scenario_number'], 4)
        self.assertFalse(candidate['leaderboard_submission'])


if __name__ == '__main__':
    unittest.main()
