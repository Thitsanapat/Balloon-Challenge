"""Experimental short-horizon timing choice for the active balloon leg.

The long route and nominal spline remain those of guarded innovation.  Only
the forecast time/reference point used by its bounded terminal correction can
move earlier along that existing spline.  Runtime inputs are observation and
public given parameters, never a seed, simulator truth, or stored trajectory.
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


def choose_active_capture_phase(
        curve, duration, elapsed, target, states, status, target_index,
        wind_profile, profile_blend, radius, max_advance, phase_step,
        max_lateral_delta, max_horizon, improvement_margin):
    """Choose an earlier existing-spline crossing by time/effort cost.

    This is a small receding-horizon approximation, not a solved orbital
    ZEM/ZEV guidance law.  If forecasts are outside the validated near-term
    range, the ordinary endpoint is retained.
    """
    values = np.asarray((duration, elapsed, profile_blend, radius,
                         max_advance, phase_step, max_lateral_delta,
                         max_horizon, improvement_margin), dtype=float)
    target = np.asarray(target, dtype=float)
    if (not np.all(np.isfinite(values)) or duration <= 0 or
            not 0 <= elapsed < duration or not 0 <= profile_blend <= 1 or
            min((radius, max_advance, phase_step, max_lateral_delta,
                 max_horizon)) <= 0 or improvement_margin < 0 or
            target.shape != (6,) or not np.all(np.isfinite(target))):
        raise ValueError("Invalid active capture phase inputs")
    remaining = duration - elapsed
    if remaining > max_horizon or remaining < 0.75:
        return remaining, False

    def cost(time_to_capture):
        ref = sample_curve(curve, duration, elapsed + time_to_capture)[0]
        xy = target[:2] + target[3:5] * time_to_capture
        xy += profile_blend * wind_profile.correction(
            states, status, target_index, time_to_capture)
        miss_xy = xy - ref[:2]
        miss_z = target[2] + target[5] * time_to_capture - ref[2]
        acceleration = 2.0 * np.linalg.norm(miss_xy) / time_to_capture**2
        # No extra vertical command is made.  Prefer a time when the existing
        # spline already crosses the balloon's vertical capture corridor.
        return (0.1 * time_to_capture / max_horizon +
                (acceleration / max_lateral_delta)**2 +
                2.0 * (miss_z / radius)**2 +
                0.6 * ((remaining - time_to_capture) / max_advance)**2)

    best_time = remaining
    baseline_cost = best_cost = cost(remaining)
    for advance in np.arange(phase_step, max_advance + 1e-8, phase_step):
        candidate = remaining - advance
        if candidate < 0.75:
            break
        value = cost(candidate)
        if value < best_cost:
            best_cost = value
            best_time = candidate
    if best_time < remaining and best_cost <= baseline_cost - improvement_margin:
        return float(best_time), True
    return float(remaining), False


class Scenario4ActivePhaseAgent(Scenario4GuardedInnovationAgent):
    def __init__(self, given_parameters, phase_max_advance=0.75,
                 phase_step=0.25, phase_max_horizon=3.0,
                 phase_improvement_margin=0.15, **kwargs):
        super().__init__(given_parameters, **kwargs)
        self.phase_max_advance = float(phase_max_advance)
        self.phase_step = float(phase_step)
        self.phase_max_horizon = float(phase_max_horizon)
        self.phase_improvement_margin = float(phase_improvement_margin)
        if (not np.all(np.isfinite((self.phase_max_advance, self.phase_step,
                                   self.phase_max_horizon,
                                   self.phase_improvement_margin))) or
                min((self.phase_max_advance, self.phase_step,
                     self.phase_max_horizon)) <= 0 or
                self.phase_improvement_margin < 0):
            raise ValueError("Invalid phase search settings")
        self.diagnostics.update(phase_advanced_steps=0,
                                phase_advance_seconds_sum=0.0)

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
                remaining = float(self.plan_start + self.plan_duration - now)
                weight = smooth_innovation_gate(
                    remaining, self.innovation_stop_seconds,
                    self.innovation_start_seconds,
                    self.innovation_fade_seconds, self.innovation_strength)
                if (weight > 0 and
                        np.linalg.norm(states[target, :3] -
                                       self.filtered_position) <=
                        self.innovation_max_target_range):
                    elapsed = float(np.clip(now - self.plan_start, 0,
                                            self.plan_duration))
                    reference = sample_curve(
                        self.plan, self.plan_duration, elapsed)[0]
                    tracking_error = float(np.linalg.norm(
                        reference - self.filtered_position))
                    if tracking_error > self.innovation_max_tracking_error:
                        self.diagnostics["guarded_skipped_tracking"] += 1
                    else:
                        capture_time, advanced = choose_active_capture_phase(
                            self.plan, self.plan_duration, elapsed,
                            states[target], states, status, target,
                            self.wind_profile, self.profile_blend,
                            self.innovation_max_tracking_error / 2.0,
                            self.phase_max_advance, self.phase_step,
                            self.innovation_max_lateral_delta,
                            self.phase_max_horizon,
                            self.phase_improvement_margin)
                        endpoint = sample_curve(
                            self.plan, self.plan_duration,
                            elapsed + capture_time)[0]
                        innovation = observed_endpoint_innovation(
                            states, status, target, capture_time, endpoint,
                            self.wind_profile, self.profile_blend)
                        if float(np.linalg.norm(innovation)) > 0.05:
                            force, saturated = bounded_innovation_force(
                                final_axis * final_magnitude, innovation,
                                capture_time, weight,
                                self.innovation_max_error,
                                self.innovation_max_lateral_delta,
                                self.innovation_time_floor, available,
                                self.max_tilt)
                            if saturated:
                                self.diagnostics["guarded_skipped_saturated"] += 1
                            else:
                                corrected_magnitude = float(np.linalg.norm(force))
                                if (np.all(np.isfinite(force)) and
                                        corrected_magnitude > 1e-9):
                                    final_axis = force / corrected_magnitude
                                    final_magnitude = corrected_magnitude
                                    self.diagnostics["guarded_applied_steps"] += 1
                                    self.diagnostics["innovation_applied_steps"] += 1
                                    self.diagnostics["innovation_error_sum"] += float(
                                        np.linalg.norm(innovation))
                                    if advanced:
                                        self.diagnostics["phase_advanced_steps"] += 1
                                        self.diagnostics["phase_advance_seconds_sum"] += (
                                            remaining - capture_time)
        self.last_desired_axis = final_axis.copy()
        return Scenario4WindProfileAgent._attitude_action(
            self, final_axis, jerk, final_magnitude, gyro, available)
