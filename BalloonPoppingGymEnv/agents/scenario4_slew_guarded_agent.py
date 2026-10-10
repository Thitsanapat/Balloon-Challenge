"""Experimental command-variation limit on the guarded endpoint correction.

Only current observations and public parameters enter this controller.  The
route, nominal spline, force envelope, and attitude controller are unchanged.
This is a development agent, not the selected qualification submission.
"""

import numpy as np

from BalloonPoppingGymEnv.agents.scenario4_forecast_innovation_agent import (
    bounded_innovation_force,
    observed_endpoint_innovation,
    smooth_innovation_gate,
)
from BalloonPoppingGymEnv.agents.scenario4_guarded_innovation_agent import (
    Scenario4GuardedInnovationAgent,
)
from BalloonPoppingGymEnv.agents.scenario4_wind_profile_agent import (
    Scenario4WindProfileAgent,
)
from BalloonPoppingGymEnv.agents.submission_sqp_v2 import sample_curve


def slew_limited_correction(previous, requested, max_rate, dt):
    """Project a horizontal force correction onto a per-step change ball."""
    previous = np.asarray(previous, dtype=float)
    requested = np.asarray(requested, dtype=float)
    if (previous.shape != (2,) or requested.shape != (2,) or
            not np.all(np.isfinite(previous)) or
            not np.all(np.isfinite(requested)) or
            not np.isfinite(max_rate) or max_rate <= 0 or
            not np.isfinite(dt) or dt <= 0):
        raise ValueError("Finite horizontal corrections and positive slew limits required")
    change = requested - previous
    length = float(np.linalg.norm(change))
    limit = max_rate * dt
    if length <= limit:
        return requested.copy(), False
    return previous + change * (limit / length), True


class Scenario4SlewGuardedAgent(Scenario4GuardedInnovationAgent):
    """Guarded innovation plus bounded variation of its additive force."""

    def __init__(self, given_parameters, innovation_slew_rate=8.0, **kwargs):
        super().__init__(given_parameters, **kwargs)
        self.innovation_slew_rate = float(innovation_slew_rate)
        if not np.isfinite(self.innovation_slew_rate) or self.innovation_slew_rate <= 0:
            raise ValueError("Positive innovation slew rate required")
        self._previous_correction = np.zeros(2)
        self._previous_correction_target = None
        self.diagnostics.update(slew_limited_steps=0, slew_applied_steps=0)

    def _attitude_action(self, axis, jerk, magnitude, gyro, available):
        path_force = np.asarray(axis, dtype=float) * float(magnitude)
        final_force = path_force.copy()
        snapshot = self._innovation_snapshot
        target = self.target_index
        if target != self._previous_correction_target:
            self._previous_correction = np.zeros(2)
            self._previous_correction_target = target
        applied = False
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
                    self.innovation_fade_seconds, self.innovation_strength)
                if (weight > 0 and
                        np.linalg.norm(states[target, :3] -
                                       self.filtered_position) <=
                        self.innovation_max_target_range):
                    elapsed = np.clip(now - self.plan_start, 0,
                                      self.plan_duration)
                    reference_position = sample_curve(
                        self.plan, self.plan_duration, elapsed)[0]
                    tracking_error = float(np.linalg.norm(
                        reference_position - self.filtered_position))
                    if tracking_error > self.innovation_max_tracking_error:
                        self.diagnostics["guarded_skipped_tracking"] += 1
                    else:
                        endpoint = sample_curve(
                            self.plan, self.plan_duration,
                            self.plan_duration)[0]
                        innovation = observed_endpoint_innovation(
                            states, status, target, time_to_go, endpoint,
                            self.wind_profile, self.profile_blend)
                        if float(np.linalg.norm(innovation)) > 0.05:
                            requested, saturated = bounded_innovation_force(
                                path_force, innovation, time_to_go, weight,
                                self.innovation_max_error,
                                self.innovation_max_lateral_delta,
                                self.innovation_time_floor, available,
                                self.max_tilt)
                            if saturated:
                                self.diagnostics["guarded_skipped_saturated"] += 1
                            else:
                                correction, limited = slew_limited_correction(
                                    self._previous_correction,
                                    requested[:2] - path_force[:2],
                                    self.innovation_slew_rate, self.dt)
                                final_force[:2] += correction
                                self._previous_correction = correction
                                self.diagnostics["slew_limited_steps"] += int(limited)
                                self.diagnostics["slew_applied_steps"] += 1
                                self.diagnostics["guarded_applied_steps"] += 1
                                self.diagnostics["innovation_applied_steps"] += 1
                                self.diagnostics["innovation_error_sum"] += float(
                                    np.linalg.norm(innovation))
                                applied = True
        if not applied:
            self._previous_correction = np.zeros(2)
        final_magnitude = float(np.linalg.norm(final_force))
        final_axis = (final_force / final_magnitude if final_magnitude > 1e-9
                      else np.asarray(axis, dtype=float))
        self.last_desired_axis = final_axis.copy()
        return Scenario4WindProfileAgent._attitude_action(
            self, final_axis, jerk, final_magnitude, gyro, available)
