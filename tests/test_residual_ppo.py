import unittest
from unittest.mock import patch
import numpy as np
from BalloonPoppingGymEnv.evaluation.evaluate import load_scenario_parameters
from BalloonPoppingGymEnv.agents.residual_ppo_agent import ResidualPPOController, ResidualPolicyAgent, residual_features, BASE_KWARGS
from BalloonPoppingGymEnv.agents.submission_time_allocation_v1 import ChainSubmissionAgent


class ResidualPPOTests(unittest.TestCase):
    def setUp(self):
        _, self.given = load_scenario_parameters(1)

    def test_zero_control_exactly_matches_frozen_attitude_action(self):
        base, residual = ChainSubmissionAgent(self.given, **BASE_KWARGS), ResidualPPOController(self.given)
        args = (np.array([.1, 0., np.sqrt(.99)]), np.array([.1, .2, 0.]), 10., np.zeros(3), 15.)
        for _ in range(10):
            for left, right in zip(base._attitude_action(*args), residual._attitude_action(*args)):
                np.testing.assert_array_equal(left, right)

    def test_features_finite_fixed_shape_only_released_targets(self):
        agent = ResidualPPOController(self.given)
        obs = dict(simulation_time=24., rocket_sensors=np.full(12, np.nan),
                   balloon_states=np.zeros((3, 6)), balloon_status=np.array([1, 0, 2]))
        before = obs['balloon_states'].copy()
        features = residual_features(agent, obs)
        self.assertEqual(features.shape, (96,))
        self.assertTrue(np.all(np.isfinite(features)))
        self.assertEqual(features[32:].reshape(8, 8)[:, 7].sum(), 1.)
        obs['balloon_states'][1:] = 10000.
        np.testing.assert_array_equal(features, residual_features(agent, obs))
        agent.set_residual([2., 0., -2.])
        np.testing.assert_array_equal(agent.residual_action, [1., 0., -1.])
        for invalid in ([np.nan, 0., 0.], [1., 2.]):
            with self.assertRaises(ValueError):
                agent.set_residual(invalid)

    def test_residual_still_respects_available_force(self):
        agent = ResidualPPOController(self.given)
        agent.set_residual([1., 1., 1.])
        with patch.object(ChainSubmissionAgent, '_attitude_action', return_value=None) as parent:
            agent._attitude_action(np.array([0., 0., 1.]), np.zeros(3), 12., np.zeros(3), 12.)
        axis, _, magnitude, _, available = parent.call_args.args
        self.assertLessEqual(magnitude, available+1e-9)
        self.assertGreaterEqual(axis[2], np.cos(agent.max_tilt)-1e-9)

    def test_deployment_holds_actions_for_exactly_ten_steps(self):
        agent = ResidualPolicyAgent(self.given, decision_steps=10)
        observation = dict(simulation_time=24., rocket_sensors=np.zeros(12))
        prefix = 'BalloonPoppingGymEnv.agents.residual_ppo_agent.'
        with patch.object(ChainSubmissionAgent, 'get_action', return_value={}), patch(
                prefix+'residual_features', return_value=np.zeros(96, np.float32)), patch.object(
                agent, 'policy_action', side_effect=([.1, .2, .3], [-.1, -.2, -.3])) as policy:
            # No policy decision before launch, or before valid telemetry.
            agent.get_action(observation)
            policy.assert_not_called()
            agent.launched = True
            observation['rocket_sensors'][0] = np.nan
            agent.get_action(observation)
            policy.assert_not_called()
            observation['rocket_sensors'][0] = 0.
            for step in range(10):
                observation['simulation_time'] = 24.01+step*agent.dt
                agent.get_action(observation)
                np.testing.assert_array_equal(agent.residual_action, [.1, .2, .3])
            self.assertEqual(policy.call_count, 1)
            observation['simulation_time'] = 24.01+10*agent.dt
            agent.get_action(observation)
            self.assertEqual(policy.call_count, 2)
            np.testing.assert_array_equal(agent.residual_action, [-.1, -.2, -.3])


if __name__ == '__main__':
    unittest.main()
