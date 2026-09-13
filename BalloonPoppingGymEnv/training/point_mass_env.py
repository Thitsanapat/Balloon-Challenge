"""Fast point-mass curriculum for scenario-1 route learning.

The environment deliberately does not replace final 6-DoF evaluation.  It
learns target ordering and a smooth acceleration policy cheaply; promising
policies must subsequently be transferred to TVC and verified in BalloonWorld.
"""

import json
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class PointMassBalloonEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, field_path, control_dt=.1, launch_time=36.3,
                 end_time=76.5, nearest_balloons=12, thrust_accel=15.,
                 hit_reward=10., shaping=.03, reference_replay=None,
                 residual_accel=3., bonus_targets=(), bonus_shaping=.2):
        super().__init__()
        with np.load(Path(field_path)) as data:
            self.balloon_flights = np.asarray(data["balloon_flights"], dtype=np.float32)
            self.release_steps = np.asarray(data["release_steps"], dtype=np.int32)
            self.field_dt = float(data["time_step"])
        self.control_dt = float(control_dt)
        self.action_repeat = int(round(self.control_dt / self.field_dt))
        if not np.isclose(self.action_repeat * self.field_dt, self.control_dt):
            raise ValueError("control_dt must be an integer multiple of field time_step")
        self.launch_step = int(round(float(launch_time) / self.field_dt))
        self.end_step = min(int(round(float(end_time) / self.field_dt)),
                            self.balloon_flights.shape[2] - 1)
        self.burnout_step = self.launch_step + int(round(30. / self.field_dt))
        self.k = int(nearest_balloons)
        self.thrust_accel = float(thrust_accel)
        self.hit_reward = float(hit_reward)
        self.shaping = float(shaping)
        self.residual_accel = float(residual_accel)
        self.residual_start = float("-inf")
        self.residual_end = float("inf")
        self.bonus_targets = tuple(int(index) for index in bonus_targets)
        self.bonus_shaping = float(bonus_shaping)
        self.bonus_distance_penalty = .02
        self.reference_times = self.reference_positions = None
        if reference_replay is not None:
            payload = json.loads(Path(reference_replay).read_text(encoding="utf-8"))
            records = payload["trajectories"] if isinstance(payload, dict) else payload
            self.reference_times = np.asarray(
                [record["time"] for record in records], dtype=float,
            )
            self.reference_positions = np.asarray([
                [np.nan if value is None else value
                 for value in record["rocket_states"][:3]]
                for record in records
            ], dtype=float)
        # Rocket position/velocity/time/fuel, then K relative position,
        # relative velocity, range and availability flag.
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(8 + self.k * 8,), dtype=np.float32,
        )
        self.action_space = spaces.Box(-1., 1., shape=(3,), dtype=np.float32)
        self.position = np.zeros(3)
        self.velocity = np.zeros(3)
        self.status = np.zeros(100, dtype=np.int8)
        self.step_index = self.launch_step
        self.previous_nearest = 0.
        self.previous_bonus_ranges = np.zeros(len(self.bonus_targets))

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.position = np.array([0., 0., 20.], dtype=np.float64)
        self.velocity = np.zeros(3, dtype=np.float64)
        self.status = np.zeros(self.balloon_flights.shape[0], dtype=np.int8)
        self.step_index = self.launch_step
        self.status[self.step_index >= self.release_steps] = 1
        self.previous_nearest = self._nearest_range()
        self.previous_bonus_ranges = self._bonus_ranges()
        return self._observation(), {"popped_count": 0}

    def _bonus_ranges(self):
        if not self.bonus_targets:
            return np.empty(0)
        indices = np.asarray(self.bonus_targets, dtype=int)
        positions = self.balloon_flights[indices, :3, self.step_index]
        return np.linalg.norm(positions - self.position, axis=1)

    @staticmethod
    def _segment_distance(a0, a1, b0, b1):
        """Closest distance between two line segments, vectorized over B."""
        u = a1 - a0
        v = b1 - b0
        w = a0 - b0
        aa = float(np.dot(u, u))
        bb = np.einsum("ij,ij->i", u[None, :], v)
        cc = np.einsum("ij,ij->i", v, v)
        dd = np.einsum("j,ij->i", u, w)
        ee = np.einsum("ij,ij->i", v, w)
        denom = aa * cc - bb * bb
        s = np.zeros(len(v))
        good = denom > 1e-12
        s[good] = np.clip((bb[good] * ee[good] - cc[good] * dd[good]) / denom[good], 0., 1.)
        t = np.zeros(len(v))
        nz = cc > 1e-12
        t[nz] = np.clip((bb[nz] * s[nz] + ee[nz]) / cc[nz], 0., 1.)
        if aa > 1e-12:
            s = np.clip((bb * t - dd) / aa, 0., 1.)
        separation = w + s[:, None] * u - t[:, None] * v
        return np.linalg.norm(separation, axis=1)

    def _nearest_range(self):
        available = self.status == 1
        if not np.any(available):
            return 300.
        positions = self.balloon_flights[:, :3, self.step_index]
        return float(np.min(np.linalg.norm(positions[available] - self.position, axis=1)))

    def _observation(self):
        positions = self.balloon_flights[:, :3, self.step_index]
        velocities = self.balloon_flights[:, 3:6, self.step_index]
        available = self.status == 1
        ranges = np.linalg.norm(positions - self.position, axis=1)
        order = np.flatnonzero(available)
        order = order[np.argsort(ranges[order])[:self.k]]
        targets = np.zeros((self.k, 8), dtype=np.float32)
        for row, index in enumerate(order):
            targets[row, :3] = (positions[index] - self.position) / 200.
            targets[row, 3:6] = (velocities[index] - self.velocity) / 50.
            targets[row, 6] = ranges[index] / 300.
            targets[row, 7] = 1.
        head = np.r_[
            self.position / np.array([300., 300., 400.]),
            self.velocity / 50.,
            (self.step_index - self.launch_step) * self.field_dt / 40.,
            max(0., self.burnout_step - self.step_index) * self.field_dt / 30.,
        ].astype(np.float32)
        return np.r_[head, targets.ravel()].astype(np.float32)

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=float), -1., 1.)
        horizontal = .95 * action[:2]
        horizontal_norm = np.linalg.norm(horizontal)
        if horizontal_norm > .95:
            horizontal *= .95 / horizontal_norm
        throttle = .5 * (action[2] + 1.)
        thrust_direction = np.r_[
            horizontal,
            np.sqrt(max(0., 1. - np.dot(horizontal, horizontal))),
        ]
        start_position = self.position.copy()
        start_step = self.step_index
        duration = min(self.action_repeat, self.end_step - self.step_index) * self.field_dt
        if self.step_index < self.burnout_step:
            burn_fraction = ((self.step_index - self.launch_step)
                             / max(1, self.burnout_step - self.launch_step))
            available_thrust = self.thrust_accel * (.88 + .26 * burn_fraction)
            if self.reference_positions is None:
                acceleration = (available_thrust * throttle * thrust_direction
                                - np.array([0., 0., 9.80665]))
            else:
                next_step = self.step_index + int(round(duration / self.field_dt))
                reference_index = int(np.argmin(np.abs(
                    self.reference_times - next_step * self.field_dt
                )))
                target = self.reference_positions[reference_index]
                baseline = 2. * (
                    target - self.position - self.velocity * duration
                ) / duration**2
                now = self.step_index * self.field_dt
                residual = (self.residual_accel * action
                            if self.residual_start <= now <= self.residual_end
                            else np.zeros(3))
                acceleration = baseline + residual
                gravity = np.array([0., 0., 9.80665])
                thrust = acceleration + gravity
                thrust_norm = np.linalg.norm(thrust)
                if thrust_norm > available_thrust:
                    thrust *= available_thrust / thrust_norm
                acceleration = thrust - gravity
        else:
            acceleration = np.array([0., 0., -9.80665])
        self.position += self.velocity * duration + .5 * acceleration * duration**2
        self.velocity += acceleration * duration
        self.step_index += int(round(duration / self.field_dt))
        self.status[self.step_index >= self.release_steps] = np.maximum(
            self.status[self.step_index >= self.release_steps], 1,
        )
        before = self.balloon_flights[:, :3, start_step]
        after = self.balloon_flights[:, :3, self.step_index]
        distances = self._segment_distance(start_position, self.position, before, after)
        hits = (self.status == 1) & (distances <= 1.5)
        count = int(np.sum(hits))
        self.status[hits] = 2
        nearest = self._nearest_range()
        reward = self.hit_reward * count + self.shaping * np.clip(
            self.previous_nearest - nearest, -5., 5.,
        ) - .002
        self.previous_nearest = nearest
        bonus_ranges = self._bonus_ranges()
        if bonus_ranges.size:
            active = np.asarray([self.status[index] == 1
                                 for index in self.bonus_targets])
            reward += self.bonus_shaping * float(np.sum(np.clip(
                self.previous_bonus_ranges[active] - bonus_ranges[active], -5., 5.,
            )))
            now = self.step_index * self.field_dt
            if self.residual_start <= now <= self.residual_end:
                reward -= self.bonus_distance_penalty * float(
                    np.sum(bonus_ranges[active])
                )
            self.previous_bonus_ranges = bonus_ranges
        terminated = self.position[2] <= 20. and self.step_index > self.launch_step + self.action_repeat
        truncated = self.step_index >= self.end_step
        return self._observation(), float(reward), terminated, truncated, {
            "popped_count": int(np.sum(self.status == 2)),
            "popped_indices": np.flatnonzero(self.status == 2).tolist(),
        }
