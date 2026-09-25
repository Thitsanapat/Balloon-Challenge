"""Training-only wrapper around an UNMODIFIED official environment.

Fresh random official field per episode. No field cache, private rocket state,
seed or future trajectory enters the policy/controller. Reward shaping is only
for training; competition score remains the official count of actual hits.
"""

import copy
import json
import time
from pathlib import Path
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from rocketpy.exceptions import InvalidParameterError
from BalloonPoppingGymEnv.envs.balloon_world import BalloonPoppingEnv
from BalloonPoppingGymEnv.evaluation.evaluate import load_scenario_parameters
from BalloonPoppingGymEnv.agents.residual_ppo_agent import ResidualPPOController, residual_features


class OfficialResidualEnv(gym.Env):
    def __init__(self, scenario=1, decision_steps=10, residual_accel=2., gamma=.995, reset_log=None):
        super().__init__()
        parameters, self.given = load_scenario_parameters(scenario)
        self.parameters = copy.deepcopy(parameters)
        self.decision_steps = int(decision_steps)
        self.residual_accel, self.gamma = float(residual_accel), float(gamma)
        if self.decision_steps < 1:
            raise ValueError('Positive decision interval required')
        self.action_space = spaces.Box(-1., 1., (3,), np.float32)
        self.observation_space = spaces.Box(-10., 10., (96,), np.float32)
        self.env = None
        self.episode_index = 0
        self.reset_log = None if reset_log is None else Path(reset_log)

    def _potential(self):
        obs = self.observation
        position = np.asarray(obs['rocket_sensors'])[6:9]
        ids = np.flatnonzero(np.asarray(obs['balloon_status']).reshape(-1) == 1)
        if not len(ids) or not np.all(np.isfinite(position)):
            return -2.
        distance = np.min(np.linalg.norm(np.asarray(obs['balloon_states'])[ids, :3]-position, axis=1))
        return -min(float(distance)/100., 2.)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.close()
        # The RNG belongs only to the external trainer, never to the agent.
        # The shipped stochastic mass distribution can sample an invalid mass.
        # Retry only this specific RESET failure; never mask flight failures or
        # alter any sampled physical value. Record every rejected seed externally.
        for attempt in range(8):
            episode_seed = int(self.np_random.integers(100_000, 2**31-1))
            parameters = copy.deepcopy(self.parameters)
            parameters['scenario']['random_seed'] = episode_seed
            self.env = BalloonPoppingEnv(render_mode=None, parameters=parameters)
            try:
                self.observation, _ = self.env.reset(seed=episode_seed)
                self.episode_wall_started = time.monotonic()
                break
            except InvalidParameterError as error:
                if not str(error).startswith('Rocket mass must be a positive number'):
                    self.close()
                    raise
                if self.reset_log is not None:
                    with self.reset_log.open('a', encoding='utf-8') as stream:
                        stream.write(json.dumps(dict(seed=episode_seed, error=str(error), attempt=attempt+1))+'\n')
                self.close()
                if attempt == 7:
                    raise
        self.agent = ResidualPPOController(copy.deepcopy(self.given), residual_accel=self.residual_accel)
        self.warmup_steps = 0
        # Same launch/planner sequence as deployment. Warmup has zero residual.
        while not (self.agent.launched and np.all(np.isfinite(self.observation['rocket_sensors']))):
            action = self.agent.get_action(self.observation)
            self.observation, _, terminated, truncated, _ = self.env.step(action)
            self.warmup_steps += 1
            if terminated or truncated:
                raise RuntimeError('Official episode ended during launch warmup')
        self.potential = self._potential()
        self.episode_index += 1
        return residual_features(self.agent, self.observation), {'warmup_steps': self.warmup_steps}

    def step(self, action):
        self.agent.set_residual(action)
        official_reward = 0.
        steps = 0
        for _ in range(self.decision_steps):
            controls = self.agent.get_action(self.observation)
            self.observation, reward, terminated, truncated, info = self.env.step(controls)
            official_reward += float(reward)
            steps += 1
            if terminated or truncated:
                break
        potential = 0. if terminated else self._potential()
        shaped = 10.*official_reward + self.gamma*potential-self.potential
        shaped -= .002*float(self.agent.residual_action@self.agent.residual_action)
        self.potential = potential
        safe_info = {'popped_count': int(info['popped_count']), 'physics_steps': steps,
                     'official_reward': official_reward, 'official_truncated': bool(truncated),
                     'episode_wall_seconds': time.monotonic()-self.episode_wall_started,
                     'simulation_time': float(self.observation['simulation_time'])}
        return residual_features(self.agent, self.observation), float(shaped), bool(terminated), bool(truncated), safe_info

    def close(self):
        if self.env is not None:
            self.env.close()
            self.env = None
