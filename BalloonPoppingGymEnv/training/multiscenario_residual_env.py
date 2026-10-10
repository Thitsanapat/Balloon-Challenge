"""Training-only mixture of released official scenarios with disjoint seed pools."""
import copy
import json
from pathlib import Path
import numpy as np
from gymnasium import spaces
from BalloonPoppingGymEnv.training.official_residual_env import OfficialResidualEnv
from BalloonPoppingGymEnv.evaluation.evaluate import load_scenario_parameters
from BalloonPoppingGymEnv.agents.multiscenario_residual_agent import (
    MultiScenarioResidualController, multiscenario_features, FEATURE_SIZE,
)

# Ranges reserved by this experiment; never forwarded as policy features.
SEED_POOLS = {'train': (1_000_000, 1_000_000_000),
              'validation': (1_000_000_000, 1_500_000_000),
              'holdout': (1_500_000_000, 2_000_000_000)}
DEFAULT_SCENARIOS = (0, 1, 2, 3, 4)
DEFAULT_WEIGHTS = (.02, .08, .15, .25, .50)


def validate_mixture(scenarios, weights):
    scenarios = tuple(scenarios)
    weights = np.asarray(weights, dtype=float)
    if (not scenarios or len(set(scenarios)) != len(scenarios)
            or any(s not in DEFAULT_SCENARIOS for s in scenarios)
            or weights.shape != (len(scenarios),) or not np.all(np.isfinite(weights))
            or np.any(weights <= 0)):
        raise ValueError('Choose unique released scenarios 0--4 with positive finite weights')
    return scenarios, weights/weights.sum()


class MultiScenarioResidualEnv(OfficialResidualEnv):
    def __init__(self, scenarios=DEFAULT_SCENARIOS, weights=DEFAULT_WEIGHTS,
                 split='train', episode_log=None, **kwargs):
        self.scenarios, self.weights = validate_mixture(scenarios, weights)
        if split not in SEED_POOLS:
            raise ValueError('Unknown seed split')
        self.split = split
        self.episode_log = Path(episode_log) if episode_log else None
        super().__init__(scenario=self.scenarios[0], **kwargs)
        self.observation_space = spaces.Box(-10., 10., (FEATURE_SIZE,), np.float32)
        self.feature_encoder = multiscenario_features
        self.active_target_shaping = True

    def _draw_episode_seed(self):
        seed = int(self.np_random.integers(*SEED_POOLS[self.split]))
        if self.episode_log:
            with self.episode_log.open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(dict(scenario=self.parameters['scenario']['number'],
                                             seed=seed, split=self.split))+'\n')
        return seed

    def _make_controller(self):
        return MultiScenarioResidualController(copy.deepcopy(self.given),
                                               residual_accel=self.residual_accel)

    def reset(self, *, seed=None, options=None):
        # Seed the trainer's RNG once, before selecting the scenario; the parent
        # reset must not re-seed it after the draw.
        if seed is not None:
            self._np_random = np.random.default_rng(seed)
        scenario = int(self.np_random.choice(self.scenarios, p=self.weights))
        parameters, self.given = load_scenario_parameters(scenario)
        self.parameters = copy.deepcopy(parameters)
        obs, info = super().reset(seed=None, options=options)
        # Metadata is for logging only, not part of the actor's observation.
        return obs, dict(info, scenario=scenario)

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        info['scenario'] = self.parameters['scenario']['number']
        return obs, reward, terminated, truncated, info
