import unittest
from unittest.mock import patch
import numpy as np
from BalloonPoppingGymEnv.agents.exit_aware_chain_agent import ExitAwareChainAgent, continuation_time
from BalloonPoppingGymEnv.evaluation.evaluate import load_scenario_parameters


class ExitAwareTests(unittest.TestCase):
    def test_velocity_toward_next_target_has_lower_cost(self):
        states = np.array([[10., 0., 20., 0., 0., 0.]])
        args = (0., 10., 12., np.radians(75), np.zeros(3))
        forward = continuation_time(states, [0], np.array([0., 0., 20.]), np.array([5., 0., 0.]), *args)
        backward = continuation_time(states, [0], np.array([0., 0., 20.]), np.array([-5., 0., 0.]), *args)
        self.assertLess(forward, backward)
        self.assertEqual(continuation_time(states, [], np.zeros(3), np.zeros(3), *args), 0.)

    def test_heuristic_caps_unreachable_and_does_not_mutate(self):
        states = np.array([[10000., 0., 20., 0., 0., 0.]])
        before = states.copy()
        value = continuation_time(states, [0], np.zeros(3), np.zeros(3), 0., 2.2, 12., 1., np.zeros(3))
        self.assertEqual(value, 2.2)
        np.testing.assert_array_equal(states, before)

    def test_disabled_exact_and_hidden_or_used_targets_excluded(self):
        _, given = load_scenario_parameters(1)
        agent = ExitAwareChainAgent(given, exit_weight=0.)
        node = ((0,), (3.,), None, np.zeros(3), np.zeros(3))
        states = np.zeros((5, 6))
        self.assertEqual(agent._node_priority(node, states, [0, 1, 2, 3], 24.), 3.)
        self.assertEqual(agent.diagnostics['exit_rank_calls'], 0)
        agent.exit_weight = .5
        agent.failed_until[2] = 30.
        with patch('BalloonPoppingGymEnv.agents.exit_aware_chain_agent.continuation_time', return_value=2.) as estimate:
            self.assertEqual(agent._node_priority(node, states, [0, 1, 2, 3], 24.), 4.)
            self.assertEqual(estimate.call_args.args[1], [1, 3])

    def test_invalid_settings(self):
        _, given = load_scenario_parameters(1)
        for settings in [dict(exit_weight=-1), dict(exit_weight=np.nan), dict(exit_horizon=.1)]:
            with self.assertRaises(ValueError):
                ExitAwareChainAgent(given, **settings)


if __name__ == '__main__':
    unittest.main()
