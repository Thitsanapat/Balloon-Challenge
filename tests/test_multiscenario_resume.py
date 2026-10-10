"""Synthetic PPO serialization test only; never a competition score test."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from scripts.resume_multiscenario_ppo import restore_policy


class SyntheticEnv(gym.Env):
    observation_space = gym.spaces.Box(-10.,10.,(112,),np.float32)
    action_space = gym.spaces.Box(-1.,1.,(3,),np.float32)
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        return np.zeros(112,np.float32), {}
    def step(self, action):
        self.steps += 1
        return np.zeros(112,np.float32), -float(action@action), self.steps >= 4, False, {}


class ResumeModelTests(unittest.TestCase):
    def test_parameters_optimizer_counters_and_new_seed(self):
        torch.set_num_threads(1)
        with TemporaryDirectory() as directory:
            env = DummyVecEnv([SyntheticEnv])
            other = DummyVecEnv([SyntheticEnv])
            try:
                model = PPO('MlpPolicy',env,device='cpu',seed=7,n_steps=8,batch_size=8,n_epochs=1,
                            policy_kwargs=dict(net_arch=[8]))
                model.learn(8)
                checkpoint = Path(directory)/'model.zip'
                model.save(checkpoint)
                restored = restore_policy(checkpoint,other,'cpu',17,8,8)
                self.assertEqual(restored.num_timesteps,8)
                self.assertEqual(restored._n_updates,model._n_updates)
                self.assertEqual(other._seeds,[17])
                self.assertEqual(restored.seed,17)
                for key,value in model.policy.state_dict().items():
                    torch.testing.assert_close(value,restored.policy.state_dict()[key],rtol=0,atol=0)
                old = model.policy.optimizer.state_dict()['state']
                new = restored.policy.optimizer.state_dict()['state']
                self.assertTrue(old)
                for key in old:
                    for name,value in old[key].items():
                        torch.testing.assert_close(value,new[key][name],rtol=0,atol=0)
                restored.learn(8,reset_num_timesteps=False)
                self.assertEqual(restored.num_timesteps,16)
                self.assertGreater(restored._n_updates,model._n_updates)
                with self.assertRaisesRegex(ValueError,'configuration'):
                    restore_policy(checkpoint,other,'cpu',17,16,8)
            finally:
                env.close()
                other.close()


if __name__ == '__main__':
    unittest.main()
