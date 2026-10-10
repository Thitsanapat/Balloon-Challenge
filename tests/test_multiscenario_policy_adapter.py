import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from BalloonPoppingGymEnv.evaluation.evaluate import load_scenario_parameters
from BalloonPoppingGymEnv.agents.multiscenario_residual_agent import FEATURE_SCHEMA, FEATURE_SIZE
from BalloonPoppingGymEnv.agents.multiscenario_ppo_policy_agent import MultiScenarioPPOPolicyAgent


class PolicyAdapterTests(unittest.TestCase):
    def test_schema_and_shapes_are_checked_before_deployment(self):
        given = load_scenario_parameters(4)[1]
        with TemporaryDirectory() as directory:
            checkpoint = Path(directory)/'checkpoint.zip'
            manifest = checkpoint.parent/'manifest.json'
            manifest.write_text(json.dumps(dict(feature_schema='old_96', source_sha256={})))
            with self.assertRaisesRegex(ValueError, 'schema'):
                MultiScenarioPPOPolicyAgent(given, checkpoint)
            manifest.write_text(json.dumps(dict(feature_schema=FEATURE_SCHEMA, source_sha256={})))
            fake = SimpleNamespace(observation_space=SimpleNamespace(shape=(96,)),
                                   action_space=SimpleNamespace(shape=(3,)))
            with patch('stable_baselines3.PPO.load', return_value=fake):
                with self.assertRaisesRegex(ValueError, 'spaces'):
                    MultiScenarioPPOPolicyAgent(given, checkpoint)
                fake.observation_space.shape = (FEATURE_SIZE,)
                a = MultiScenarioPPOPolicyAgent(given, checkpoint)
                self.assertFalse(a.joint_refresh_allow_shortening)
                np.testing.assert_array_equal(a.residual_action, np.zeros(3))

    def test_controller_overrides_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'overrides'):
            MultiScenarioPPOPolicyAgent(load_scenario_parameters(4)[1], 'unused.zip', launch_time=12.)


if __name__ == '__main__':
    unittest.main()
