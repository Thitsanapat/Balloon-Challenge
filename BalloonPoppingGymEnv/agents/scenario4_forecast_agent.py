"""Short-horizon balloon motion forecast from live observations only.

The ordinary chain planner assumes each observed balloon keeps its current
velocity until interception. Scenario 4 has altitude-dependent gusts, so this
experimental agent estimates recent velocity change for each released balloon.
It supplies a small, bounded position correction to the existing planner only
when a balloon could be intercepted soon. The estimator never reads the
simulator, its random seed, or a precomputed balloon trajectory.
"""

import numpy as np

from BalloonPoppingGymEnv.agents.time_allocation_agent import TimeAllocationAgent


class BalloonAccelerationTracker:
    """Track local balloon acceleration using observed velocity differences."""

    def __init__(
        self,
        count,
        smoothing_tau=0.25,
        min_history=0.15,
        max_sample_gap=0.10,
        max_horizontal_acceleration=2.5,
        max_vertical_acceleration=3.0,
        max_position_correction=2.5,
    ):
        settings = (
            smoothing_tau,
            min_history,
            max_sample_gap,
            max_horizontal_acceleration,
            max_vertical_acceleration,
            max_position_correction,
        )
        if (count < 1 or not np.all(np.isfinite(settings))
                or min(settings) < 0 or max_sample_gap == 0):
            raise ValueError("Invalid balloon forecast settings")
        self.count = int(count)
        self.smoothing_tau = float(smoothing_tau)
        self.min_history = float(min_history)
        self.max_sample_gap = float(max_sample_gap)
        self.max_horizontal_acceleration = float(max_horizontal_acceleration)
        self.max_vertical_acceleration = float(max_vertical_acceleration)
        self.max_position_correction = float(max_position_correction)
        self.last_time = None
        self.last_velocity = np.zeros((self.count, 3))
        self.last_valid = np.zeros(self.count, dtype=bool)
        self.acceleration = np.zeros((self.count, 3))
        self.history = np.zeros(self.count)

    def update(self, observation):
        now = float(observation["simulation_time"])
        states = np.asarray(observation["balloon_states"], dtype=float)
        status = np.asarray(observation["balloon_status"]).reshape(-1)
        if (not np.isfinite(now) or states.shape != (self.count, 6)
                or status.shape != (self.count,)):
            raise ValueError("Invalid balloon observation shape or time")
        if self.last_time is not None and now <= self.last_time:
            return

        valid = (status == 1) & np.all(np.isfinite(states), axis=1)
        dt = None if self.last_time is None else now - self.last_time
        paired = (valid & self.last_valid) if dt is not None and dt <= self.max_sample_gap else np.zeros(self.count, dtype=bool)
        self.acceleration[~paired] = 0.0
        self.history[~paired] = 0.0
        if np.any(paired):
            rate = (states[paired, 3:6] - self.last_velocity[paired]) / dt
            horizontal = np.linalg.norm(rate[:, :2], axis=1)
            horizontal_scale = np.minimum(1.0, self.max_horizontal_acceleration / np.maximum(horizontal, 1e-12))
            rate[:, :2] *= horizontal_scale[:, None]
            rate[:, 2] = np.clip(rate[:, 2], -self.max_vertical_acceleration, self.max_vertical_acceleration)
            gain = 1.0 if self.smoothing_tau == 0 else -np.expm1(-dt / self.smoothing_tau)
            self.acceleration[paired] += gain * (rate - self.acceleration[paired])
            self.history[paired] += dt

        self.last_velocity[valid] = states[valid, 3:6]
        self.last_valid = valid
        self.last_time = now

    def position_correction(self, index, horizon):
        """Return a clipped 0.5*a*t² correction, or zero without recent data."""
        index = int(index)
        horizon = float(horizon)
        if (index < 0 or index >= self.count or not np.isfinite(horizon)
                or horizon <= 0 or not self.last_valid[index]
                or self.history[index] < self.min_history):
            return np.zeros(3)
        offset = 0.5 * self.acceleration[index] * horizon ** 2
        length = np.linalg.norm(offset)
        if length > self.max_position_correction:
            offset *= self.max_position_correction / length
        return offset


class Scenario4ForecastAgent(TimeAllocationAgent):
    """Use local observed balloon acceleration in short-range route planning."""

    def __init__(
        self,
        given_parameters,
        forecast_horizon=1.5,
        forecast_max_range=30.0,
        forecast_nominal_speed=6.0,
        forecast_smoothing_tau=0.25,
        forecast_min_history=0.15,
        forecast_max_sample_gap=0.10,
        forecast_max_horizontal_acceleration=2.5,
        forecast_max_vertical_acceleration=3.0,
        forecast_max_position_correction=2.5,
        **kwargs,
    ):
        super().__init__(given_parameters, **kwargs)
        self.forecast_horizon = float(forecast_horizon)
        self.forecast_max_range = float(forecast_max_range)
        self.forecast_nominal_speed = float(forecast_nominal_speed)
        if (not np.all(np.isfinite([self.forecast_horizon, self.forecast_max_range, self.forecast_nominal_speed]))
                or min(self.forecast_horizon, self.forecast_max_range, self.forecast_nominal_speed) <= 0):
            raise ValueError("Invalid forecast range or horizon")
        self.forecast_tracker = BalloonAccelerationTracker(
            given_parameters["balloon"]["num"],
            smoothing_tau=forecast_smoothing_tau,
            min_history=forecast_min_history,
            max_sample_gap=forecast_max_sample_gap,
            max_horizontal_acceleration=forecast_max_horizontal_acceleration,
            max_vertical_acceleration=forecast_max_vertical_acceleration,
            max_position_correction=forecast_max_position_correction,
        )
        self.diagnostics.update(forecast_plans=0, forecast_corrected_targets=0)

    def get_action(self, observation):
        # Sample once per environment step. Launch-direction trials may call
        # _make_plan repeatedly, but must all see the same observation history.
        self.forecast_tracker.update(observation)
        return super().get_action(observation)

    def _make_plan(self, observation, position, velocity, now):
        states = np.asarray(observation["balloon_states"], dtype=float)
        status = np.asarray(observation["balloon_status"]).reshape(-1)
        position = np.asarray(position, dtype=float)
        velocity = np.asarray(velocity, dtype=float)
        adjusted = None
        corrected = 0
        active = self.route[0] if self.route else None

        for index in np.flatnonzero(status == 1):
            relative = states[index, :3] - position
            distance = float(np.linalg.norm(relative))
            if not np.isfinite(distance) or distance > self.forecast_max_range:
                continue
            closing_speed = float(np.dot(relative, velocity - states[index, 3:6]) / max(distance, 1e-9))
            horizon = distance / max(self.forecast_nominal_speed, closing_speed)
            if index == active and self.deadlines:
                horizon = float(self.deadlines[0] - now)
            if not 0 < horizon <= self.forecast_horizon:
                continue
            correction = self.forecast_tracker.position_correction(index, horizon)
            if not np.any(correction):
                continue
            if adjusted is None:
                adjusted = states.copy()
            adjusted[index, :3] += correction
            corrected += 1

        if corrected:
            observation = dict(observation, balloon_states=adjusted)
            self.diagnostics["forecast_plans"] += 1
            self.diagnostics["forecast_corrected_targets"] += corrected
        return super()._make_plan(observation, position, velocity, now)
