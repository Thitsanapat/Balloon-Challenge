"""Deployment must retain the training wrapper's ten-step decision schedule."""
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import numpy as np
from BalloonPoppingGymEnv.agents.multiscenario_ppo_policy_agent import MultiScenarioPPOPolicyAgent


class PolicyCadenceTests(unittest.TestCase):
    def test_launch_warmup_and_target_changes_do_not_shift_decision_clock(self):
        agent = object.__new__(MultiScenarioPPOPolicyAgent)
        agent.launched = False
        agent.dt = .01
        agent.next_policy = 0.
        agent.residual_action = np.zeros(3)
        prediction = Mock(return_value=(np.array([.1, -.2, .3]), None))
        agent.policy = SimpleNamespace(predict=prediction)
        prefix = 'BalloonPoppingGymEnv.agents.multiscenario_ppo_policy_agent.'
        with patch(prefix+'multiscenario_features', return_value=np.zeros(112)), \
                patch(prefix+'MultiScenarioResidualController.get_action', return_value='control'):
            observation = dict(simulation_time=42., rocket_sensors=np.zeros(12))
            self.assertEqual(agent.get_action(observation), 'control')
            prediction.assert_not_called()
            agent.launched = True
            observation['rocket_sensors'][:] = np.nan
            agent.get_action(observation)
            prediction.assert_not_called()
            observation['rocket_sensors'][:] = 0.
            for step in range(21):
                observation['simulation_time'] = 42.01 + step*.01
                # The parent clears stale corrections after a target changes;
                # this must not trigger an additional policy decision.
                if step == 3:
                    agent.set_residual(np.zeros(3))
                agent.get_action(observation)
            self.assertEqual(prediction.call_count, 3)
            self.assertTrue(all(c.kwargs['deterministic'] for c in prediction.call_args_list))
            np.testing.assert_array_equal(agent.residual_action, [.1, -.2, .3])


if __name__ == '__main__':
    unittest.main()
