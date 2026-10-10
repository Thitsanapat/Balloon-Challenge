"""Small terminal miss correction on top of the Scenario-4 wind-profile route.

Only given parameters and current/past observations are used.  The baseline
quintic route, PD tracking loop and attitude-rate feedforward remain intact.
The correction predicts the tracking error *remaining under that PD loop*;
adding the full current error again would double-count baseline feedback.
"""

import numpy as np

from BalloonPoppingGymEnv.agents.scenario4_wind_profile_agent import (
    Scenario4WindProfileAgent,
)
from BalloonPoppingGymEnv.agents.submission_sqp_v2 import (
    allocate_acceleration,
    sample_curve,
)


def critically_damped_tracking_residual(position_error, velocity_error,
                                         frequency, horizon):
    """Residual error after the baseline's ideal e''+2*w*e'+w²*e=0 loop."""
    position_error = np.asarray(position_error, dtype=float)
    velocity_error = np.asarray(velocity_error, dtype=float)
    if (position_error.shape != (3,) or velocity_error.shape != (3,) or
            not np.all(np.isfinite(position_error)) or
            not np.all(np.isfinite(velocity_error)) or
            not np.isfinite(frequency) or frequency <= 0 or
            not np.isfinite(horizon) or horizon < 0):
        raise ValueError("Invalid tracking residual inputs")
    return ((1.0 + frequency * horizon) * position_error +
            horizon * velocity_error) * np.exp(-frequency * horizon)


def terminal_miss_vector(states, status, target_index, horizon,
                         nominal_endpoint, tracking_residual,
                         wind_profile, profile_blend):
    """Observed target forecast minus predicted rocket endpoint, in 3-D."""
    states = np.asarray(states, dtype=float)
    status = np.asarray(status).reshape(-1)
    endpoint = np.asarray(nominal_endpoint, dtype=float)
    residual = np.asarray(tracking_residual, dtype=float)
    if (states.ndim != 2 or states.shape[1] != 6 or
            status.shape != (len(states),) or
            not 0 <= target_index < len(states) or
            status[target_index] != 1 or not np.isfinite(horizon) or
            horizon <= 0 or endpoint.shape != (3,) or residual.shape != (3,) or
            not np.all(np.isfinite(states[target_index])) or
            not np.all(np.isfinite(endpoint)) or
            not np.all(np.isfinite(residual)) or
            not np.isfinite(profile_blend) or not 0 <= profile_blend <= 1):
        return np.zeros(3)
    forecast = states[target_index, :3] + states[target_index, 3:6] * horizon
    forecast[:2] += profile_blend * wind_profile.correction(
        states, status, target_index, horizon)
    return forecast - endpoint - residual


def feasible_terminal_force(base_force, miss, horizon, weight, deadband,
                            max_error, max_delta, max_vertical_delta,
                            time_floor, available, max_tilt):
    """Add only the feasible part of a bounded 3-D endpoint correction."""
    base = np.asarray(base_force, dtype=float)
    miss = np.asarray(miss, dtype=float)
    limits = (horizon, weight, deadband, max_error, max_delta,
              max_vertical_delta, time_floor, available, max_tilt)
    if (base.shape != (3,) or miss.shape != (3,) or
            not np.all(np.isfinite(base)) or not np.all(np.isfinite(miss)) or
            not np.all(np.isfinite(limits)) or horizon <= 0 or
            not 0 <= weight <= 1 or deadband < 0 or max_error <= deadband or
            max_delta <= 0 or max_vertical_delta <= 0 or time_floor <= 0 or
            available < 0 or not 0 <= max_tilt < np.pi/2):
        raise ValueError("Invalid terminal force inputs")
    distance = float(np.linalg.norm(miss))
    if weight == 0 or distance <= deadband:
        return base.copy(), False
    error = miss * min(max_error, distance - deadband) / distance
    correction = 2.0 * weight * error / max(horizon, time_floor)**2
    correction[2] = np.clip(correction[2], -max_vertical_delta,
                            max_vertical_delta)
    norm = float(np.linalg.norm(correction))
    if norm > max_delta:
        correction *= max_delta / norm
    if np.linalg.norm(correction) < 1e-8:
        return base.copy(), False

    def feasible(scale):
        candidate = base + scale * correction
        allocated = allocate_acceleration(candidate, available, max_tilt)
        return np.allclose(candidate, allocated, atol=1e-5, rtol=0)

    if feasible(1.0):
        return base + correction, False
    low, high = 0.0, 1.0
    for _ in range(18):
        middle = (low + high) / 2
        if feasible(middle):
            low = middle
        else:
            high = middle
    if low < 1e-4:
        return base.copy(), True
    return base + low * correction, True


def terminal_gate(horizon, stop, start, fade, strength):
    """Smoothly enter and leave a short correction window."""
    values = (horizon, stop, start, fade, strength)
    if (not np.all(np.isfinite(values)) or stop < 0 or start <= stop or
            fade <= 0 or 2*fade > start-stop or not 0 <= strength <= 1):
        raise ValueError("Invalid terminal gate")
    if not stop < horizon < start:
        return 0.0
    def smoothstep(x):
        x = float(np.clip(x, 0, 1))
        return x*x*(3-2*x)
    return float(strength * min(smoothstep((horizon-stop)/fade),
                                smoothstep((start-horizon)/fade)))


class Scenario4TerminalResidualAgent(Scenario4WindProfileAgent):
    """Use a bounded predicted miss, while leaving the parent route untouched."""

    def __init__(self, given_parameters, terminal_start=3.5,
                 terminal_stop=0.3, terminal_fade=0.55,
                 terminal_strength=0.35, terminal_max_range=90.0,
                 terminal_deadband=0.65, terminal_max_error=3.0,
                 terminal_max_delta=1.5, terminal_max_vertical_delta=0.8,
                 terminal_time_floor=1.0, **kwargs):
        super().__init__(given_parameters, **kwargs)
        self.terminal_start = float(terminal_start)
        self.terminal_stop = float(terminal_stop)
        self.terminal_fade = float(terminal_fade)
        self.terminal_strength = float(terminal_strength)
        self.terminal_max_range = float(terminal_max_range)
        self.terminal_deadband = float(terminal_deadband)
        self.terminal_max_error = float(terminal_max_error)
        self.terminal_max_delta = float(terminal_max_delta)
        self.terminal_max_vertical_delta = float(terminal_max_vertical_delta)
        self.terminal_time_floor = float(terminal_time_floor)
        terminal_gate((self.terminal_start+self.terminal_stop)/2,
                      self.terminal_stop, self.terminal_start,
                      self.terminal_fade, self.terminal_strength)
        if (not np.all(np.isfinite((self.terminal_max_range,
                self.terminal_deadband, self.terminal_max_error,
                self.terminal_max_delta, self.terminal_max_vertical_delta,
                self.terminal_time_floor))) or
                self.terminal_max_range <= 0 or self.terminal_deadband < 0 or
                self.terminal_max_error <= self.terminal_deadband or
                self.terminal_max_delta <= 0 or
                self.terminal_max_vertical_delta <= 0 or
                self.terminal_time_floor <= 0):
            raise ValueError("Invalid terminal correction limits")
        self._terminal_snapshot = None
        self.diagnostics.update(terminal_applied_steps=0,
                                terminal_limited_steps=0,
                                terminal_predicted_miss_sum=0.0)

    def get_action(self, observation):
        self._terminal_snapshot = (
            float(observation['simulation_time']),
            np.asarray(observation['balloon_status']).reshape(-1).copy(),
            np.asarray(observation['balloon_states'], dtype=float).copy())
        try:
            return super().get_action(observation)
        finally:
            self._terminal_snapshot = None

    def _attitude_action(self, axis, jerk, magnitude, gyro, available):
        snapshot = self._terminal_snapshot
        target = self.target_index
        if (snapshot is not None and target is not None and
                self.plan is not None and self.plan_duration > 0 and
                self.filtered_position is not None and
                self.filtered_velocity is not None):
            now, status, states = snapshot
            if (0 <= target < len(status) and status[target] == 1 and
                    np.all(np.isfinite(states[target]))):
                horizon = float(self.plan_start + self.plan_duration - now)
                weight = terminal_gate(horizon, self.terminal_stop,
                                       self.terminal_start, self.terminal_fade,
                                       self.terminal_strength)
                if (weight > 0 and np.linalg.norm(states[target, :3] -
                        self.filtered_position) <= self.terminal_max_range):
                    elapsed = float(np.clip(now - self.plan_start, 0,
                                            self.plan_duration))
                    nominal_p, nominal_v, _, _ = sample_curve(
                        self.plan, self.plan_duration, elapsed)
                    endpoint = sample_curve(self.plan, self.plan_duration,
                                            self.plan_duration)[0]
                    residual = critically_damped_tracking_residual(
                        self.filtered_position - nominal_p,
                        self.filtered_velocity - nominal_v,
                        self.tracking_frequency, horizon)
                    miss = terminal_miss_vector(
                        states, status, target, horizon, endpoint, residual,
                        self.wind_profile, self.profile_blend)
                    force, limited = feasible_terminal_force(
                        np.asarray(axis) * magnitude, miss, horizon,
                        weight, self.terminal_deadband,
                        self.terminal_max_error, self.terminal_max_delta,
                        self.terminal_max_vertical_delta,
                        self.terminal_time_floor, available, self.max_tilt)
                    corrected_magnitude = float(np.linalg.norm(force))
                    if (corrected_magnitude > 1e-9 and
                            not np.array_equal(force, np.asarray(axis)*magnitude)):
                        axis = force / corrected_magnitude
                        magnitude = corrected_magnitude
                        self.diagnostics['terminal_applied_steps'] += 1
                        self.diagnostics['terminal_predicted_miss_sum'] += float(
                            np.linalg.norm(miss))
                    self.diagnostics['terminal_limited_steps'] += int(limited)
        self.last_desired_axis = np.asarray(axis).copy()
        return super()._attitude_action(axis, jerk, magnitude, gyro, available)
