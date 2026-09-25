import unittest
from unittest.mock import patch
import numpy as np
from BalloonPoppingGymEnv.agents.reachability_chain_agent import reachability_order, ReachabilityChainAgent
from BalloonPoppingGymEnv.agents.final_approach_agent import FinalApproachAgent


class ReachabilityTests(unittest.TestCase):
    def test_uses_velocity_and_only_candidate_ids_without_mutation(self):
        states = np.array([[20., 0., 20., 0., 0., 0.], [-5., 0., 20., 0., 0., 0.],
                           [0., 0., 20., 0., 0., 0.]])
        before = states.copy()
        order = reachability_order(states, [0, 1], np.array([0., 0., 20.]),
                                   np.array([10., 0., 0.]), 0., 10., 15., np.radians(75.))
        self.assertEqual(order, [0, 1])
        np.testing.assert_array_equal(states, before)
        self.assertEqual(reachability_order(states, [], np.zeros(3), np.zeros(3), 0., 10., 15., 1.), [])

    def test_disabled_and_released_failed_filter(self):
        agent = ReachabilityChainAgent.__new__(ReachabilityChainAgent)
        agent.nearest_fraction, agent.branch_targets = 1., 2
        agent.failed_until = {1: 5.}
        agent.launch_time, agent.burn_time, agent.max_tilt = 0., 30., 1.
        agent.available_acceleration = lambda t: 15.
        with patch.object(FinalApproachAgent, '_candidate_ids', return_value=[0, 3]):
            args = (np.zeros((5, 6)), [0, 1, 2, 3], (2,), 0., np.zeros(3), np.zeros(3), 0.)
            self.assertEqual(agent._candidate_ids(*args), [0, 3])
            agent.nearest_fraction = 0.
            self.assertEqual(set(agent._candidate_ids(*args)), {0, 3})


if __name__ == '__main__':
    unittest.main()
