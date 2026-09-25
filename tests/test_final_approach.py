import unittest
from unittest.mock import patch
import numpy as np
from BalloonPoppingGymEnv.agents.final_approach_agent import FinalApproachAgent
from BalloonPoppingGymEnv.agents.chain_launch_agent import ChainLaunchAgent


class FinalApproachTests(unittest.TestCase):
    def agent(self):
        agent = FinalApproachAgent.__new__(FinalApproachAgent)
        agent.plan, agent.target_index = np.zeros((6, 3)), 0
        agent.plan_start, agent.plan_duration = 0., 10.
        agent.final_approach_guard, agent.next_plan = .15, 10.3
        agent.diagnostics = dict(final_approach_holds=0)
        return agent

    def test_hold_before_deadline_but_review_expired_or_popped_target(self):
        for now, status, expected in ((9.88, 1, 0), (9.7, 1, 1), (10., 1, 1), (9.88, 2, 1)):
            agent = self.agent()
            with patch.object(ChainLaunchAgent, '_make_plan') as parent:
                agent._make_plan(dict(balloon_status=[status]), np.zeros(3), np.zeros(3), now)
                self.assertEqual(parent.call_count, expected)
            self.assertEqual(agent.plan_duration, 10.)

    def test_review_is_due_at_deadline_not_later(self):
        agent = self.agent()
        action = {'launch': True}
        with patch.object(ChainLaunchAgent, 'get_action', return_value=action):
            self.assertIs(agent.get_action(dict(simulation_time=9.88)), action)
        self.assertEqual(agent.next_plan, 10.)

    def test_disabled_guard_is_exact_passthrough(self):
        agent = self.agent()
        agent.final_approach_guard = 0.
        with patch.object(ChainLaunchAgent, '_make_plan') as parent:
            agent._make_plan(dict(balloon_status=[1]), np.zeros(3), np.zeros(3), 9.99)
            parent.assert_called_once()
        with patch.object(ChainLaunchAgent, 'get_action', return_value={}):
            agent.get_action(dict(simulation_time=9.99))
        self.assertEqual(agent.next_plan, 10.3)


if __name__ == '__main__':
    unittest.main()
