import unittest

import numpy as np

from BalloonPoppingGymEnv.agents.scenario4_margin_learned_joint_agent import (
    Scenario4MarginLearnedJointAgent,
)


class MarginLearnedJointTests(unittest.TestCase):
    def test_weak_actuator_margin_preserves_physical_ranking(self):
        agent = object.__new__(Scenario4MarginLearnedJointAgent)
        agent.launched = True
        agent.risk_penalty_seconds = 1.0
        agent.minimum_margin = 0.10
        agent._risk_planning_state = (np.zeros(3), np.zeros(3), 42.0)
        agent.diagnostics = {'risk_margin_skips': 0}
        agent._first_leg_margin = lambda node, now: 0.05
        node = ((0,), (2.0,), np.zeros((1, 6, 3)), None, None)
        self.assertEqual(agent._node_priority(node, np.zeros((1, 6)),
                                              np.array([0]), 42.0), 2.0)
        self.assertEqual(agent.diagnostics['risk_margin_skips'], 1)


if __name__ == '__main__':
    unittest.main()
