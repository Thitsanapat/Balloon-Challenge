"""Scenario 4 residual-policy interface on the validated wind-profile agent.

Zero residual reproduces the established controller exactly.  A trained policy
can only request a bounded acceleration correction; the existing allocator
retains the published thrust/tilt limits.  The policy sees observation-derived
features, never the seed or hidden simulator state.
"""

import numpy as np

from BalloonPoppingGymEnv.agents.residual_ppo_agent import residual_features
from BalloonPoppingGymEnv.agents.scenario4_wind_profile_agent import (
    Scenario4WindProfileAgent,
)
from BalloonPoppingGymEnv.agents.submission_sqp_v2 import allocate_acceleration


SCENARIO4_BASE_KWARGS = dict(
    launch_time=42.0, launch_candidates=3, replan_interval=0.4,
    beam_width=12, search_depth=14, branch_targets=8, time_grid=0.5,
    terminal_weight=0.0, capture_fraction=0.0, reserve=0.8,
    max_tilt=75.0, max_axis_rate=1.2, attitude_frequency=4.0,
    timing_candidates=4, timing_iterations=20, timing_safety=0.0,
    timing_only_more_hits=True, profile_horizon=8.0,
    profile_max_range=150.0, profile_blend=0.75,
    profile_max_position_correction=12.0,
)


def scenario4_residual_features(agent, observation):
    """Keep the currently pursued balloon in the fixed-size feature set."""
    return residual_features(agent, observation, active_first=True)


class Scenario4ResidualController(Scenario4WindProfileAgent):
    """Bounded correction atop the selected Scenario 4 controller."""

    def __init__(self, given_parameters, residual_accel=2.0, **kwargs):
        settings = dict(SCENARIO4_BASE_KWARGS)
        settings.update(kwargs)
        super().__init__(given_parameters, **settings)
        self.residual_accel = float(residual_accel)
        if not np.isfinite(self.residual_accel) or self.residual_accel < 0:
            raise ValueError("Invalid residual bound")
        self.residual_action = np.zeros(3)

    def set_residual(self, action):
        value = np.asarray(action, dtype=float)
        if value.shape != (3,) or not np.all(np.isfinite(value)):
            raise ValueError("Expected three finite residual actions")
        self.residual_action = np.clip(value, -1.0, 1.0).copy()

    def _make_plan(self, observation, position, velocity, now):
        previous_target = self.target_index
        result = super()._make_plan(observation, position, velocity, now)
        if self.target_index != previous_target:
            self.set_residual(np.zeros(3))
        return result

    def _attitude_action(self, axis, jerk, magnitude, gyro, available):
        if (self.plan is not None and available > 0 and self.residual_accel
                and np.any(self.residual_action)):
            requested = axis * magnitude + self.residual_accel * self.residual_action
            force = allocate_acceleration(requested, available, self.max_tilt)
            magnitude = float(np.linalg.norm(force))
            axis = (force / max(magnitude, 1e-9) if magnitude
                    else np.array([0.0, 0.0, 1.0]))
            self.last_desired_axis = axis.copy()
        return super()._attitude_action(axis, jerk, magnitude, gyro, available)


class Scenario4ResidualPolicyAgent(Scenario4ResidualController):
    """Deployment interface; exported weights may replace policy_action."""

    def __init__(self, given_parameters, decision_steps=10, **kwargs):
        super().__init__(given_parameters, **kwargs)
        self.decision_steps = int(decision_steps)
        if self.decision_steps < 1:
            raise ValueError("Positive decision interval required")
        self.next_policy = 0.0

    def policy_action(self, features):
        return np.zeros(3)

    def get_action(self, observation):
        now = float(observation["simulation_time"])
        status = np.asarray(observation["balloon_status"]).reshape(-1)
        active = self.target_index
        if (self.plan is None or active is None or
                not 0 <= active < len(status) or status[active] != 1):
            self.set_residual(np.zeros(3))
            self.next_policy = now
        if (self.launched and
                self.plan is not None and active is not None and
                0 <= active < len(status) and status[active] == 1 and
                np.all(np.isfinite(observation["rocket_sensors"])) and
                now + 1e-8 >= self.next_policy):
            self.set_residual(self.policy_action(
                scenario4_residual_features(self, observation)))
            self.next_policy = now + self.decision_steps * self.dt
        return super().get_action(observation)
