import unittest
from unittest.mock import patch

import numpy as np

from BalloonPoppingGymEnv.agents.scenario4_pure_joint_refresh_agent import (
    Scenario4PureJointRefreshAgent,
)


class PureJointRefreshTests(unittest.TestCase):
    def agent(self):
        agent = Scenario4PureJointRefreshAgent.__new__(
            Scenario4PureJointRefreshAgent)
        agent.route = [0, 1]
        agent.deadlines = [2.0, 4.0]
        agent.launch_time = 0.0
        agent.burn_time = 100.0
        agent.acceleration = np.zeros(3)
        agent.terminal_weight = 0.0
        agent.diagnostics = dict(pure_joint_attempts=0,
                                 pure_joint_accepted=0)
        agent.ends_v, agent.ends_a = [], []
        agent._valid = lambda curves, times, starts, samples: (
            np.ones(2, dtype=bool), None,
            np.ones((2, 3)), np.zeros((2, 3)))
        agent._select = lambda index, curve, duration, now: setattr(
            agent, 'selected', (index, duration, now))
        return agent

    def observation(self):
        return dict(balloon_status=np.array([1, 1]),
                    balloon_states=np.array([
                        [1., 2., 3., 4., 5., 6.],
                        [7., 8., 9., 1., 2., 3.],
                    ]))

    def test_keeps_ids_and_deadlines_when_repair_is_feasible(self):
        agent = self.agent()
        with patch(
                'BalloonPoppingGymEnv.agents.scenario4_pure_joint_refresh_agent.free_chain',
                return_value=np.zeros((2, 6, 3))) as chain:
            accepted = agent._rescue_committed_leg(
                self.observation(), np.zeros(3), np.zeros(3), 0.0, 2.0)
        self.assertTrue(accepted)
        self.assertEqual(agent.route, [0, 1])
        self.assertEqual(agent.deadlines, [2.0, 4.0])
        self.assertEqual(agent.selected, (0, 2.0, 0.0))
        self.assertEqual(agent.diagnostics['pure_joint_accepted'], 1)
        np.testing.assert_allclose(chain.call_args.args[3][0], [9., 12., 15.])

    def test_invalid_active_target_does_not_change_plan(self):
        agent = self.agent()
        observation = self.observation()
        observation['balloon_status'][0] = 2
        self.assertFalse(agent._rescue_committed_leg(
            observation, np.zeros(3), np.zeros(3), 0.0, 2.0))
        self.assertEqual(agent.deadlines, [2.0, 4.0])
        self.assertEqual(agent.diagnostics['pure_joint_attempts'], 0)


if __name__ == '__main__':
    unittest.main()
