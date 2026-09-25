"""Official-environment training wrapper for high-level target selection.

The wrapper receives fresh external seeds. Policies see a fixed vector made
from public observations and choose a rank within released candidates. The
unchanged chain controller executes the selected first leg and all GNC commands.
"""

import copy
import json
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from rocketpy.exceptions import InvalidParameterError

from BalloonPoppingGymEnv.agents.route_selector_agent import (
    SELECTOR_FEATURES, SelectorChainAgent,
)
from BalloonPoppingGymEnv.envs.balloon_world import BalloonPoppingEnv
from BalloonPoppingGymEnv.evaluation.evaluate import load_scenario_parameters


class OfficialSelectorEnv(gym.Env):
    metadata = {'render_modes': []}

    def __init__(self, scenario=1, selector_candidates=8, decision_steps=40,
                 gamma=.995, reset_log=None, record_queries=False,
                 selector_enabled=True):
        super().__init__()
        parameters, self.given = load_scenario_parameters(scenario)
        self.parameters = copy.deepcopy(parameters)
        self.selector_candidates = int(selector_candidates)
        self.decision_steps, self.gamma = int(decision_steps), float(gamma)
        if not 1 <= self.selector_candidates <= 16 or self.decision_steps < 1:
            raise ValueError('Invalid selector environment limits')
        size = self.selector_candidates*SELECTOR_FEATURES + 4
        self.action_space = spaces.Discrete(self.selector_candidates)
        self.observation_space = spaces.Box(-10., 10., (size,), dtype=np.float32)
        self.reset_log = None if reset_log is None else Path(reset_log)
        self.record_queries = bool(record_queries)
        self.selector_enabled = bool(selector_enabled)
        self.queries = []
        self.env = None

    def _potential(self):
        status = np.asarray(self.observation['balloon_status']).reshape(-1)
        ids = np.flatnonzero(status == 1)
        position = np.asarray(self.observation['rocket_sensors'])[6:9]
        if not len(ids) or not np.all(np.isfinite(position)):
            return -2.
        states = np.asarray(self.observation['balloon_states'])
        return -min(float(np.min(np.linalg.norm(states[ids, :3]-position, axis=1)))/100., 2.)

    def _features(self):
        features, mask, _ = self.agent.current_selector_features(self.observation)
        return np.clip(np.nan_to_num(features), -10., 10.).astype(np.float32), mask

    def action_masks(self):
        _, mask = self._features()
        # Maskable PPO requires at least one selectable action while no target is
        # released; action zero then has no effect on the parent controller.
        if not np.any(mask):
            mask[0] = True
        return mask

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.close()
        self.queries = []
        for attempt in range(8):
            episode_seed = int(self.np_random.integers(100_000, 2**31-1))
            parameters = copy.deepcopy(self.parameters)
            parameters['scenario']['random_seed'] = episode_seed
            self.env = BalloonPoppingEnv(render_mode=None, parameters=parameters)
            try:
                self.observation, _ = self.env.reset(seed=episode_seed)
                break
            except InvalidParameterError as error:
                if not str(error).startswith('Rocket mass must be a positive number'):
                    self.close()
                    raise
                if self.reset_log is not None:
                    with self.reset_log.open('a', encoding='utf-8') as stream:
                        stream.write(json.dumps({'seed': episode_seed, 'attempt': attempt+1,
                                                 'error': str(error)})+'\n')
                self.close()
                if attempt == 7:
                    raise
        self.agent = SelectorChainAgent(copy.deepcopy(self.given),
                                        selector_candidates=self.selector_candidates,
                                        selector_enabled=self.selector_enabled)
        self.episode_wall_started = time.monotonic()
        while not (self.agent.launched and np.all(np.isfinite(self.observation['rocket_sensors']))):
            controls = self.agent.get_action(self.observation)
            self.observation, _, terminated, truncated, _ = self.env.step(controls)
            if terminated or truncated:
                raise RuntimeError('Official episode ended during launch warmup')
        self.potential = self._potential()
        return self._features()[0], {'warmup_complete': True}

    def step(self, action):
        features, mask = self._features()
        action = int(action)
        valid = 0 <= action < self.selector_candidates and bool(mask[action])
        self.agent.set_selector_action(action if valid else 0)
        before_events = len(self.agent.selector_events)
        official_reward, steps = 0., 0
        for _ in range(self.decision_steps):
            controls = self.agent.get_action(self.observation)
            self.observation, reward, terminated, truncated, info = self.env.step(controls)
            official_reward += float(reward)
            steps += 1
            if terminated or truncated:
                break
        potential = 0. if terminated else self._potential()
        reward = 10.*official_reward + self.gamma*potential-self.potential
        self.potential = potential
        outcome = {'popped_count': int(info['popped_count']), 'physics_steps': steps,
                   'official_reward': official_reward, 'official_truncated': bool(truncated),
                   'simulation_time': float(self.observation['simulation_time']),
                   'episode_wall_seconds': time.monotonic()-self.episode_wall_started}
        if self.record_queries and np.any(mask):
            # Keep only decisions that actually reached the parent root search.
            # A high-level action during an already committed leg has no causal
            # target-selection effect, so it must not become a training label.
            self.queries.append({'features': features.copy(), 'mask': mask.copy(),
                                 'action': action if valid else 0,
                                 'root_selected': len(self.agent.selector_events) > before_events})
        return self._features()[0], float(reward), bool(terminated), bool(truncated), outcome

    def close(self):
        if self.env is not None:
            self.env.close()
            self.env = None
