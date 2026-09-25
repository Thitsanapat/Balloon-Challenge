"""Observation-only residual control interface; no replay or field loading.

Training supplies a held residual action. Deployment supplies a learned policy
using the exact same feature encoder and decision interval. The planner remains
the frozen, reproduced 10-point source; residuals cannot increase thrust limits.
"""

import numpy as np
from BalloonPoppingGymEnv.agents.submission_time_allocation_v1 import (
    ChainSubmissionAgent, sample_curve, allocate_acceleration, _rotate_body_to_world,
)


BASE_KWARGS = dict(launch_time=24., launch_candidates=3, replan_interval=.4,
    beam_width=12, search_depth=14, branch_targets=8, time_grid=.5,
    terminal_weight=0., capture_fraction=0., max_tilt=75., max_axis_rate=1.2,
    attitude_frequency=4., timing_candidates=4, timing_iterations=20,
    timing_safety=0., timing_only_more_hits=True)


def residual_features(agent, observation, nearest=8):
    sensors = np.asarray(observation['rocket_sensors'], dtype=float)
    now = float(observation['simulation_time'])
    position = np.nan_to_num(sensors[6:9], nan=0.).copy()
    velocity = np.nan_to_num(sensors[9:12], nan=0.)
    gyro = np.nan_to_num(sensors[:3], nan=0.)
    if not np.all(np.isfinite(sensors[6:9])):
        position = np.array([0., 0., agent.elevation])
    ep, ev = np.zeros(3), np.zeros(3)
    if agent.plan is not None:
        p, v, _, _ = sample_curve(agent.plan, agent.plan_duration,
                                  np.clip(now-agent.plan_start, 0., agent.plan_duration))
        ep, ev = p-position, v-velocity
    local_position = position-np.array([0., 0., agent.elevation])
    head = np.r_[local_position/300., velocity/40., gyro/2.,
        _rotate_body_to_world(agent.quaternion, [0., 0., 1.]),
        np.clip((agent.launch_time+agent.burn_time-now)/agent.burn_time, 0., 1.),
        agent.previous_tvc/agent.control['max_gimbal_angle'], agent.previous_throttle,
        ep/10., ev/10., agent.last_desired_axis, agent.available_acceleration(now)/20.,
        agent.disturbance/3., agent.residual_action]
    states = np.asarray(observation['balloon_states'], dtype=float)
    ids = np.flatnonzero(np.asarray(observation['balloon_status']).reshape(-1) == 1)
    distances = np.linalg.norm(states[ids, :3]-position, axis=1)
    ids = ids[np.argsort(distances, kind='stable')[:nearest]]
    targets = np.zeros((nearest, 8))
    if len(ids):
        targets[:len(ids), :3] = (states[ids, :3]-position)/100.
        targets[:len(ids), 3:6] = (states[ids, 3:6]-velocity)/25.
        targets[:len(ids), 6] = np.linalg.norm(states[ids, :3]-position, axis=1)/150.
        targets[:len(ids), 7] = 1.
    return np.clip(np.nan_to_num(np.r_[head, targets.ravel()]), -10., 10.).astype(np.float32)


class ResidualPPOController(ChainSubmissionAgent):
    def __init__(self, given_parameters, residual_accel=2., **kwargs):
        settings = dict(BASE_KWARGS)
        settings.update(kwargs)
        super().__init__(given_parameters, **settings)
        self.residual_accel = float(residual_accel)
        if not np.isfinite(self.residual_accel) or self.residual_accel < 0:
            raise ValueError('Invalid residual bound')
        self.residual_action = np.zeros(3)

    def set_residual(self, action):
        value = np.asarray(action, dtype=float)
        if value.shape != (3,) or not np.all(np.isfinite(value)):
            raise ValueError('Expected three finite residual actions')
        self.residual_action = np.clip(value, -1., 1.).copy()

    def _attitude_action(self, axis, jerk, magnitude, gyro, available):
        # Exact bypass matters: zero residual must reproduce the frozen agent.
        if available > 0 and self.residual_accel and np.any(self.residual_action):
            force = allocate_acceleration(axis*magnitude+self.residual_accel*self.residual_action,
                                          available, self.max_tilt)
            magnitude = float(np.linalg.norm(force))
            axis = force/max(magnitude, 1e-9) if magnitude else np.array([0., 0., 1.])
            self.last_desired_axis = axis.copy()
        return super()._attitude_action(axis, jerk, magnitude, gyro, available)


class ResidualPolicyAgent(ResidualPPOController):
    """Subclass with policy_action(features); the final export embeds weights."""
    def __init__(self, given_parameters, decision_steps=10, **kwargs):
        super().__init__(given_parameters, **kwargs)
        self.decision_steps = int(decision_steps)
        if self.decision_steps < 1:
            raise ValueError('Positive decision interval required')
        self.next_policy = 0.

    def policy_action(self, features):
        return np.zeros(3)

    def get_action(self, observation):
        now = float(observation['simulation_time'])
        if (self.launched and np.all(np.isfinite(observation['rocket_sensors']))
                and now+1e-8 >= self.next_policy):
            self.set_residual(self.policy_action(residual_features(self, observation)))
            self.next_policy = now+self.decision_steps*self.dt
        return super().get_action(observation)
