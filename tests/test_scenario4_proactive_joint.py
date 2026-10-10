import unittest
from unittest.mock import patch

import numpy as np

from BalloonPoppingGymEnv.agents.scenario4_proactive_joint_agent import Scenario4ProactiveJointAgent
from BalloonPoppingGymEnv.agents.scenario4_pure_joint_refresh_agent import Scenario4PureJointRefreshAgent
from BalloonPoppingGymEnv.evaluation.evaluate import load_scenario_parameters


class ProactiveJointTests(unittest.TestCase):
    def agent(self):
        _, given = load_scenario_parameters(4)
        agent = Scenario4ProactiveJointAgent(given, launch_candidates=0, timing_candidates=0)
        agent.route, agent.deadlines, agent.target_index = [0, 1], [5., 8.], 0
        agent.plan = np.zeros((6, 3))
        agent.plan[-1] = [8., 0., 60.]
        agent._feasible = lambda *args: True
        agent._select = lambda *args: setattr(agent, 'selected', args)
        return agent

    def curve(self):
        curve = np.zeros((6, 3))
        curve[-1] = [10., 0., 60.]
        return curve

    def test_failed_joint_attempt_retains_valid_refresh_and_deadlines(self):
        agent = self.agent()
        curve = self.curve()
        self.assertFalse(agent._accept_committed_curve(curve, 3., 2.))
        with patch.object(Scenario4PureJointRefreshAgent, '_rescue_committed_leg', return_value=False):
            self.assertTrue(agent._rescue_committed_leg({}, np.zeros(3), np.zeros(3), 2., 3.))
        np.testing.assert_array_equal(agent.selected[1], curve)
        self.assertEqual(agent.route, [0, 1])
        self.assertEqual(agent.deadlines, [5., 8.])
        self.assertEqual(agent.diagnostics['proactive_fallbacks'], 1)
        self.assertIsNone(agent._pending_feasible_refresh)

    def test_successful_joint_attempt_does_not_restore_old_curve(self):
        agent = self.agent()
        self.assertFalse(agent._accept_committed_curve(self.curve(), 3., 2.))
        with patch.object(Scenario4PureJointRefreshAgent, '_rescue_committed_leg', return_value=True):
            self.assertTrue(agent._rescue_committed_leg({}, np.zeros(3), np.zeros(3), 2., 3.))
        self.assertFalse(hasattr(agent, 'selected'))
        self.assertEqual(agent.diagnostics['proactive_accepted'], 1)

    def test_small_shift_single_leg_and_late_approach_keep_original_behavior(self):
        agent = self.agent()
        self.assertTrue(agent._accept_committed_curve(agent.plan.copy(), 3., 2.))
        self.assertTrue(agent._accept_committed_curve(self.curve(), 0.5, 2.))
        agent.route = [0]
        self.assertTrue(agent._accept_committed_curve(self.curve(), 3., 2.))
        self.assertEqual(agent.diagnostics['proactive_requests'], 0)

    def test_original_infeasibility_does_not_gain_an_invalid_fallback(self):
        agent = self.agent()
        agent._feasible = lambda *args: False
        self.assertFalse(agent._accept_committed_curve(self.curve(), 3., 2.))
        with patch.object(Scenario4PureJointRefreshAgent, '_rescue_committed_leg', return_value=False):
            self.assertFalse(agent._rescue_committed_leg({}, np.zeros(3), np.zeros(3), 2., 3.))
        self.assertFalse(hasattr(agent, 'selected'))


if __name__ == '__main__':
    unittest.main()
