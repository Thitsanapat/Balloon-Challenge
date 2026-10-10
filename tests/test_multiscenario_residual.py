import unittest
from unittest.mock import patch
import numpy as np
from BalloonPoppingGymEnv.evaluation.evaluate import load_scenario_parameters
from BalloonPoppingGymEnv.agents.multiscenario_residual_agent import (
    MultiScenarioResidualController, multiscenario_features, FEATURE_SIZE,
)
from BalloonPoppingGymEnv.agents.scenario4_joint_refresh_agent import Scenario4JointRefreshAgent
from BalloonPoppingGymEnv.agents.scenario4_residual_agent import SCENARIO4_BASE_KWARGS
from BalloonPoppingGymEnv.training.multiscenario_residual_env import (
    MultiScenarioResidualEnv, SEED_POOLS, validate_mixture,
)
from scripts.train_multiscenario_residual_ppo import source_hashes
from tests.test_official_residual_env import FakeAgent, FakeEnv


class MultiScenarioTests(unittest.TestCase):
    def test_feature_schema_is_common_and_finite_for_all_released_scenarios(self):
        for scenario in range(5):
            given = load_scenario_parameters(scenario)[1]
            a = MultiScenarioResidualController(given)
            obs = dict(simulation_time=0., rocket_sensors=np.full(12, np.nan),
                       balloon_states=np.zeros((10, 6)), balloon_status=np.zeros(10))
            features = multiscenario_features(a, obs)
            self.assertEqual(features.shape, (FEATURE_SIZE,))
            self.assertTrue(np.all(np.isfinite(features)))
            self.assertFalse(a.joint_refresh_allow_shortening)
            self.assertNotIn('scenario', a.given_parameters)

    def test_zero_residual_preserves_full_route_attitude(self):
        given = load_scenario_parameters(4)[1]
        plain = Scenario4JointRefreshAgent(given, joint_refresh_allow_shortening=False,
                                          **SCENARIO4_BASE_KWARGS)
        mixed = MultiScenarioResidualController(given)
        args = (np.array([0., 0., 1.]), np.zeros(3), 11., np.zeros(3), 13.)
        for _ in range(5):
            for left, right in zip(plain._attitude_action(*args), mixed._attitude_action(*args)):
                np.testing.assert_array_equal(left, right)
        self.assertIs(mixed._rescue_committed_leg.__func__, plain._rescue_committed_leg.__func__)

    def test_seed_pools_disjoint_and_mixture_validation(self):
        ranges = list(SEED_POOLS.values())
        for i, (lo, hi) in enumerate(ranges):
            for lo2, hi2 in ranges[i+1:]:
                self.assertTrue(hi <= lo2 or hi2 <= lo)
        for scenarios, weights in [([4, 4], [1, 1]), ([5], [1]), ([4], [0]), ([4], [np.nan])]:
            with self.assertRaises(ValueError):
                validate_mixture(scenarios, weights)

    def test_reset_selects_released_scenarios_deterministically_without_leaking_seed(self):
        prefix = 'BalloonPoppingGymEnv.training.'
        sequences = []
        with patch(prefix+'official_residual_env.BalloonPoppingEnv', FakeEnv), \
                patch(prefix+'multiscenario_residual_env.MultiScenarioResidualController', FakeAgent), \
                patch(prefix+'multiscenario_residual_env.multiscenario_features',
                      return_value=np.zeros(FEATURE_SIZE, np.float32)):
            for _ in range(2):
                env = MultiScenarioResidualEnv(scenarios=[2, 3, 4], weights=[1, 1, 2])
                sequence = []
                for episode in range(6):
                    obs, info = env.reset(seed=123 if episode == 0 else None)
                    sequence.append((info['scenario'], FakeEnv.seeds[-1]))
                    self.assertEqual(obs.shape, (FEATURE_SIZE,))
                    self.assertNotIn('seed', info)
                    self.assertTrue(1_000_000 <= FakeEnv.seeds[-1] < 1_000_000_000)
                env.close()
                sequences.append(sequence)
        self.assertEqual(sequences[0], sequences[1])
        self.assertGreater(len({s for s, _ in sequences[0]}), 1)

    def test_hash_manifest_includes_all_scenarios_and_shared_controller(self):
        hashes = source_hashes()
        for scenario in range(5):
            self.assertIn(f'BalloonPoppingGymEnv/envs/scenario_parameters/scenario_{scenario}_parameters.yaml', hashes)
        self.assertIn('BalloonPoppingGymEnv/agents/scenario4_joint_refresh_agent.py', hashes)


if __name__ == '__main__':
    unittest.main()
