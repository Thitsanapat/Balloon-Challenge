"""Experimental terminal correction from live target forecast innovation.

The existing wind-profile planner chooses the route and nominal spline. This
controller compares its active spline endpoint with a fresh forecast made only
from the current released-balloon observations. A bounded horizontal force
corrects that difference as the target approaches. No simulator internals,
seed information, or prerecorded path enters the policy.
"""

import numpy as np

from BalloonPoppingGymEnv.agents.scenario4_wind_profile_agent import (
    Scenario4WindProfileAgent,
)
from BalloonPoppingGymEnv.agents.submission_sqp_v2 import (
    allocate_acceleration,
    sample_curve,
)


def smooth_innovation_gate(time_to_go, stop, start, fade, strength):
    """Turn a terminal correction on and off continuously in time."""
    values = np.asarray((time_to_go, stop, start, fade, strength), dtype=float)
    if (not np.all(np.isfinite(values)) or stop < 0 or start <= stop or
            fade <= 0 or 2 * fade > start - stop or
            not 0 <= strength <= 1):
        raise ValueError("Invalid forecast-innovation time window")
    if time_to_go <= stop or time_to_go >= start:
        return 0.0

    def smoothstep(value):
        x = float(np.clip(value, 0.0, 1.0))
        return x * x * (3.0 - 2.0 * x)

    return float(strength * min(
        smoothstep((time_to_go - stop) / fade),
        smoothstep((start - time_to_go) / fade),
    ))


def observed_endpoint_innovation(states, status, index, time_to_go,
                                 nominal_endpoint, wind_profile,
                                 profile_blend):
    """Observed target endpoint minus the active spline's nominal endpoint.

    The profile term is the same fraction used by the route planner. If the
    current observations still agree with that plan, innovation is exactly
    zero, so guidance does not disturb the ordinary spline controller.
    """
    states = np.asarray(states, dtype=float)
    status = np.asarray(status).reshape(-1)
    nominal_endpoint = np.asarray(nominal_endpoint, dtype=float)
    if (states.ndim != 2 or states.shape[1] != 6 or
            status.shape != (len(states),) or not 0 <= index < len(states) or
            status[index] != 1 or nominal_endpoint.shape != (3,) or
            not np.all(np.isfinite(states[index])) or
            not np.all(np.isfinite(nominal_endpoint)) or
            not np.isfinite(time_to_go) or time_to_go <= 0 or
            not np.isfinite(profile_blend) or not 0 <= profile_blend <= 1):
        return np.zeros(2)
    target = states[index]
    forecast_xy = target[:2] + target[3:5] * time_to_go
    forecast_xy += profile_blend * wind_profile.correction(
        states, status, index, time_to_go)
    return forecast_xy - nominal_endpoint[:2]


def bounded_innovation_force(path_force, innovation, time_to_go, weight,
                             max_innovation, max_lateral_delta,
                             time_floor, available, max_tilt):
    """Apply the forecast error within public thrust and tilt limits."""
    path_force = np.asarray(path_force, dtype=float)
    innovation = np.asarray(innovation, dtype=float)
    scalars = np.asarray((time_to_go, weight, max_innovation,
                          max_lateral_delta, time_floor, available,
                          max_tilt), dtype=float)
    if (path_force.shape != (3,) or innovation.shape != (2,) or
            not np.all(np.isfinite(path_force)) or
            not np.all(np.isfinite(innovation)) or
            not np.all(np.isfinite(scalars)) or time_to_go <= 0 or
            not 0 <= weight <= 1 or max_innovation <= 0 or
            max_lateral_delta <= 0 or time_floor <= 0 or available < 0 or
            not 0 <= max_tilt < np.pi / 2):
        raise ValueError("Invalid forecast-innovation force input")
    # A tiny measured endpoint difference is dominated by observation noise;
    # passing the original force through also preserves nominal feedforward.
    if weight == 0 or float(np.linalg.norm(innovation)) <= 0.05:
        return path_force.copy(), False

    error_norm = float(np.linalg.norm(innovation))
    bounded_error = innovation * min(1.0, max_innovation / error_norm)
    correction = 2.0 * weight * bounded_error / max(time_to_go, time_floor)**2
    correction_norm = float(np.linalg.norm(correction))
    if correction_norm > max_lateral_delta:
        correction *= max_lateral_delta / correction_norm
    requested = path_force.copy()
    requested[:2] += correction
    allocated = allocate_acceleration(requested, available, max_tilt)
    saturated = float(np.linalg.norm(allocated - requested)) > 0.1
    return allocated, saturated


class Scenario4ForecastInnovationAgent(Scenario4WindProfileAgent):
    """Correct live forecast drift without changing target selection."""

    def __init__(self, given_parameters, innovation_start_seconds=5.0,
                 innovation_stop_seconds=0.2,
                 innovation_fade_seconds=0.6,
                 innovation_strength=0.65,
                 innovation_max_target_range=150.0,
                 innovation_max_error=6.0,
                 innovation_max_lateral_delta=4.0,
                 innovation_time_floor=1.0, **kwargs):
        super().__init__(given_parameters, **kwargs)
        self.innovation_start_seconds = float(innovation_start_seconds)
        self.innovation_stop_seconds = float(innovation_stop_seconds)
        self.innovation_fade_seconds = float(innovation_fade_seconds)
        self.innovation_strength = float(innovation_strength)
        self.innovation_max_target_range = float(innovation_max_target_range)
        self.innovation_max_error = float(innovation_max_error)
        self.innovation_max_lateral_delta = float(
            innovation_max_lateral_delta)
        self.innovation_time_floor = float(innovation_time_floor)
        smooth_innovation_gate(
            0.5 * (self.innovation_start_seconds +
                   self.innovation_stop_seconds),
            self.innovation_stop_seconds, self.innovation_start_seconds,
            self.innovation_fade_seconds, self.innovation_strength)
        bounds = (self.innovation_max_target_range,
                  self.innovation_max_error,
                  self.innovation_max_lateral_delta,
                  self.innovation_time_floor)
        if not np.all(np.isfinite(bounds)) or min(bounds) <= 0:
            raise ValueError("Invalid forecast-innovation limits")
        self._innovation_snapshot = None
        self.diagnostics.update(innovation_applied_steps=0,
                                innovation_saturated_steps=0,
                                innovation_error_sum=0.0)

    def get_action(self, observation):
        self._innovation_snapshot = (
            float(observation["simulation_time"]),
            np.asarray(observation["balloon_status"]).reshape(-1).copy(),
            np.asarray(observation["balloon_states"], dtype=float).copy(),
        )
        try:
            return super().get_action(observation)
        finally:
            self._innovation_snapshot = None

    def _attitude_action(self, axis, jerk, magnitude, gyro, available):
        final_axis = np.asarray(axis, dtype=float)
        final_magnitude = float(magnitude)
        snapshot = self._innovation_snapshot
        target = self.target_index
        if (snapshot is not None and target is not None and
                self.plan is not None and self.plan_duration > 0 and
                self.filtered_position is not None):
            now, status, states = snapshot
            if (0 <= target < len(status) and status[target] == 1 and
                    np.all(np.isfinite(states[target])) and
                    np.all(np.isfinite(self.filtered_position))):
                time_to_go = float(self.plan_start + self.plan_duration - now)
                weight = smooth_innovation_gate(
                    time_to_go, self.innovation_stop_seconds,
                    self.innovation_start_seconds,
                    self.innovation_fade_seconds,
                    self.innovation_strength)
                if (weight > 0 and
                        np.linalg.norm(states[target, :3] -
                                       self.filtered_position) <=
                        self.innovation_max_target_range):
                    nominal_endpoint = sample_curve(
                        self.plan, self.plan_duration,
                        self.plan_duration)[0]
                    innovation = observed_endpoint_innovation(
                        states, status, target, time_to_go,
                        nominal_endpoint, self.wind_profile,
                        self.profile_blend)
                    if float(np.linalg.norm(innovation)) > 0.05:
                        force, saturated = bounded_innovation_force(
                            final_axis * final_magnitude, innovation,
                            time_to_go, weight, self.innovation_max_error,
                            self.innovation_max_lateral_delta,
                            self.innovation_time_floor, available,
                            self.max_tilt)
                        corrected_magnitude = float(np.linalg.norm(force))
                        if (np.all(np.isfinite(force)) and
                                corrected_magnitude > 1e-9):
                            final_axis = force / corrected_magnitude
                            final_magnitude = corrected_magnitude
                            self.diagnostics["innovation_applied_steps"] += 1
                            self.diagnostics[
                                "innovation_saturated_steps"] += int(saturated)
                            self.diagnostics["innovation_error_sum"] += float(
                                np.linalg.norm(innovation))
        self.last_desired_axis = final_axis.copy()
        return super()._attitude_action(
            final_axis, jerk, final_magnitude, gyro, available)
