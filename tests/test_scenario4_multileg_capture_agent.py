import unittest
from types import SimpleNamespace

import numpy as np
import yaml

from BalloonPoppingGymEnv.agents.scenario4_multileg_capture_agent import (
    Scenario4MultiLegCaptureAgent,
    multileg_failure_probability,
)


class MultiLegCaptureTests(unittest.TestCase):
    def node(self):
        # Constant-position splines suffice to verify leg starts and horizons.
        c0, c1 = np.zeros((6, 3)), np.zeros((6, 3))
        c1[0] = [4., 0., 2.]
        return ((0, 1), (2., 3.), (c0, c1), None, None)

    def test_discounted_risk_uses_projected_later_target(self):
        calls = []

        class RecordingModel:
            def probability(self, features):
                calls.append(features)
                return (0.2, 0.8)[len(calls) - 1]

        states = np.array([[2., 0., 5., 0., 0., 0.],
                           [8., 0., 5., 1., 0., 0.]])
        risk = multileg_failure_probability(
            self.node(), states, np.zeros(3), np.zeros(3), 42.,
            lambda now: 20., RecordingModel())
        self.assertAlmostEqual(risk, (0.8 + 0.5 * 0.2) / 1.5)
        self.assertEqual(len(calls), 2)
        # The second target is projected two seconds and starts from c1[0].
        self.assertAlmostEqual(calls[1][1], np.linalg.norm([6., 0., 3.]) / 150.)

    def test_priority_falls_back_before_launch(self):
        agent = object.__new__(Scenario4MultiLegCaptureAgent)
        agent.launched = False
        agent.risk_penalty_seconds = 0.5
        agent._risk_planning_state = None
        self.assertEqual(agent._node_priority(self.node(), np.zeros((2, 6)),
                                              [0, 1], 42.), 5.0)

    def test_candidate_config_keeps_selected_control_settings(self):
        from pathlib import Path
        root = Path(__file__).resolve().parents[1]
        selected = yaml.safe_load((root / 'BalloonPoppingGymEnv/evaluation/configs/submission_scenario4_guarded_innovation_launch42.yaml').read_text())
        candidate = yaml.safe_load((root / 'BalloonPoppingGymEnv/evaluation/configs/scenario4_multileg_capture_launch42.yaml').read_text())
        for key, value in selected['agent_kwargs'].items():
            self.assertEqual(candidate['agent_kwargs'][key], value, key)
        self.assertFalse(candidate['leaderboard_submission'])


if __name__ == '__main__':
    unittest.main()
