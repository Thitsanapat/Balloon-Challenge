import unittest
from unittest.mock import patch
import numpy as np
from BalloonPoppingGymEnv.training.official_residual_env import OfficialResidualEnv


class FakeAgent:
    def __init__(self, given, **kwargs):
        self.launched = False
        self.residual_action = np.zeros(3)
        assert 'scenario' not in given

    def get_action(self, observation):
        assert set(observation) == {'rocket_sensors', 'balloon_states', 'balloon_status', 'simulation_time'}
        self.launched = True
        return {}

    def set_residual(self, action):
        self.residual_action = np.asarray(action)


class FakeEnv:
    seeds = []
    def __init__(self, **kwargs):
        self.steps = 0
        self.closed = False

    def obs(self):
        return dict(rocket_sensors=np.zeros(12), balloon_states=np.array([[10., 0., 0., 0., 0., 0.]]),
                    balloon_status=np.ones(1), simulation_time=self.steps*.01)

    def reset(self, seed=None):
        FakeEnv.seeds.append(seed)
        return self.obs(), {'private_state': object()}

    def step(self, action):
        self.steps += 1
        return self.obs(), 0., False, self.steps >= 4, {'popped_count': 0, 'private_state': object()}

    def close(self):
        self.closed = True


class OfficialResidualEnvTests(unittest.TestCase):
    def test_external_seed_is_not_in_policy_info_and_truncation_stops_repeat(self):
        prefix = 'BalloonPoppingGymEnv.training.official_residual_env.'
        with patch(prefix+'BalloonPoppingEnv', FakeEnv), patch(prefix+'ResidualPPOController', FakeAgent), patch(
                prefix+'residual_features', return_value=np.zeros(96, np.float32)):
            env = OfficialResidualEnv(decision_steps=10)
            env.reset(seed=300)
            first = env.env
            first_seed = FakeEnv.seeds[-1]
            obs, reward, terminated, truncated, info = env.step(np.zeros(3))
            self.assertFalse(terminated)
            self.assertTrue(truncated)
            self.assertEqual(info['physics_steps'], 3)
            self.assertNotIn('private_state', info)
            self.assertNotIn('seed', info)
            self.assertTrue(np.isfinite(reward))
            env.reset()
            self.assertTrue(first.closed)
            self.assertNotEqual(first_seed, FakeEnv.seeds[-1])
            self.assertGreaterEqual(FakeEnv.seeds[-1], 100000)
            env.close()

    def test_invalid_mass_reset_is_retried_but_other_errors_are_not(self):
        from rocketpy.exceptions import InvalidParameterError
        class InvalidMassOnce(FakeEnv):
            attempts = 0
            def reset(self, seed=None):
                InvalidMassOnce.attempts += 1
                if InvalidMassOnce.attempts == 1:
                    raise InvalidParameterError('Rocket mass must be a positive number, got -0.1.')
                return super().reset(seed)
        prefix = 'BalloonPoppingGymEnv.training.official_residual_env.'
        with patch(prefix+'BalloonPoppingEnv', InvalidMassOnce), patch(prefix+'ResidualPPOController', FakeAgent), patch(
                prefix+'residual_features', return_value=np.zeros(96, np.float32)):
            env = OfficialResidualEnv()
            env.reset(seed=300)
            self.assertEqual(InvalidMassOnce.attempts, 2)
            env.close()
        with patch(prefix+'BalloonPoppingEnv', FakeEnv), patch.object(FakeEnv, 'reset', side_effect=ValueError('unexpected')):
            env = OfficialResidualEnv()
            with self.assertRaisesRegex(ValueError, 'unexpected'):
                env.reset(seed=300)
            env.close()


if __name__ == '__main__':
    unittest.main()
