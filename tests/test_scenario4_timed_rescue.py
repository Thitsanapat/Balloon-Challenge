import unittest
from unittest.mock import patch

import numpy as np

from BalloonPoppingGymEnv.agents.scenario4_pure_joint_refresh_agent import (
    Scenario4PureJointRefreshAgent,
)
from BalloonPoppingGymEnv.agents.scenario4_timed_rescue_agent import (
    Scenario4TimedRescueAgent,
)


class TimedRescueTests(unittest.TestCase):
    def make_agent(self):
        agent = Scenario4TimedRescueAgent.__new__(Scenario4TimedRescueAgent)
        agent.route, agent.deadlines = [0, 1], [3.0, 5.0]
        agent.launch_time, agent.burn_time = 0.0, 10.0
        agent.acceleration = np.zeros(3)
        agent.terminal_weight = 0.0
        agent.timed_rescue_delays = (0.4, 0.8)
        agent.diagnostics = dict(timed_rescue_attempts=0,
                                 timed_rescue_accepted=0,
                                 timed_rescue_delay_sum=0.0)
        agent._select = lambda index, curve, duration, now: setattr(
            agent, 'selected', (index, duration, now))
        return agent

    def observation(self):
        return dict(balloon_status=np.ones(2, dtype=int),
                    balloon_states=np.array([[1., 0., 2., 1., 0., 1.],
                                             [3., 0., 4., 0., 0., 1.]]))

    def test_accepts_first_feasible_delay_and_preserves_target_order(self):
        agent = self.make_agent()
        # Only the 0.8 s alternative is feasible; all route legs are checked.
        agent._valid = lambda curves, times, starts, samples: (
            np.array([times[0] >= 3.8] * 2), None,
            np.ones((2, 3)), np.zeros((2, 3)))
        with patch.object(Scenario4PureJointRefreshAgent, '_rescue_committed_leg',
                          return_value=False), patch(
                    'BalloonPoppingGymEnv.agents.scenario4_timed_rescue_agent.free_chain',
                    return_value=np.zeros((2, 6, 3))):
            self.assertTrue(agent._rescue_committed_leg(
                self.observation(), np.zeros(3), np.zeros(3), 0.0, 3.0))
        self.assertEqual(agent.route, [0, 1])
        np.testing.assert_allclose(agent.deadlines, [3.8, 5.8])
        self.assertEqual(agent.selected, (0, 3.8, 0.0))
        self.assertEqual(agent.diagnostics['timed_rescue_attempts'], 2)

    def test_missing_target_cannot_change_deadlines(self):
        agent = self.make_agent()
        observation = self.observation()
        observation['balloon_status'][0] = 2
        with patch.object(Scenario4PureJointRefreshAgent, '_rescue_committed_leg',
                          return_value=False):
            self.assertFalse(agent._rescue_committed_leg(
                observation, np.zeros(3), np.zeros(3), 0.0, 3.0))
        self.assertEqual(agent.deadlines, [3.0, 5.0])
        self.assertEqual(agent.diagnostics['timed_rescue_attempts'], 0)


if __name__ == '__main__':
    unittest.main()
