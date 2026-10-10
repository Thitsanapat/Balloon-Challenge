"""Local validation adapter for a trusted mixed-scenario PPO checkpoint.

Loads learned weights once, not simulation trajectories. This multi-file
adapter is for fresh official validation, not the leaderboard's single-file
submission packer. Supply checkpoint, manifest, and matching source together.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
from BalloonPoppingGymEnv.agents.multiscenario_residual_agent import (
    MultiScenarioResidualController, multiscenario_features, FEATURE_SCHEMA, FEATURE_SIZE,
)


class MultiScenarioPPOPolicyAgent(MultiScenarioResidualController):
    def __init__(self, given_parameters, checkpoint, **kwargs):
        from stable_baselines3 import PPO
        # The pilot uses a fixed residual bound/decision cadence. Controller
        # tuning after training would change the MDP; keep kwargs empty here.
        if kwargs:
            raise ValueError('Pilot validation does not allow controller overrides')
        super().__init__(given_parameters)
        checkpoint = Path(checkpoint).resolve()
        manifest = json.loads((checkpoint.parent/'manifest.json').read_text(encoding='utf-8'))
        if manifest.get('feature_schema') != FEATURE_SCHEMA:
            raise ValueError('Incompatible policy feature schema')
        root = Path(__file__).resolve().parents[2]
        for name, digest in manifest['source_sha256'].items():
            if name.startswith('BalloonPoppingGymEnv/agents/'):
                if hashlib.sha256((root/name).read_bytes()).hexdigest() != digest:
                    raise ValueError(f'Training controller source differs: {name}')
        self.policy = PPO.load(checkpoint, device='cpu')
        if self.policy.observation_space.shape != (FEATURE_SIZE,) or self.policy.action_space.shape != (3,):
            raise ValueError('Incompatible checkpoint observation/action spaces')
        self.next_policy = 0.

    def get_action(self, observation):
        now = float(observation['simulation_time'])
        if (self.launched and np.all(np.isfinite(observation['rocket_sensors']))
                and now + 1e-8 >= self.next_policy):
            action, _ = self.policy.predict(multiscenario_features(self, observation), deterministic=True)
            self.set_residual(action)
            self.next_policy = now + 10 * self.dt
        return super().get_action(observation)
