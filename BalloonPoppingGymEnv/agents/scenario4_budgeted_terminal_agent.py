"""Experimental partial terminal correction within the available thrust cone.

The selected agent drops a terminal innovation when its full requested force
would saturate. This variant uses the largest feasible fraction instead. It
does not read a seed, environment object, simulator truth, or hidden trajectory.
The reviewed standalone submission agent is not changed.
"""

import numpy as np

from BalloonPoppingGymEnv.agents.submission_scenario4_guarded_innovation_v1 import (
    Scenario4GuardedInnovationAgent, Scenario4SubmissionAgent,
    bounded_innovation_force, observed_endpoint_innovation, sample_curve,
    smooth_innovation_gate,
)


def largest_feasible_innovation(path_force, innovation, time_to_go, weight,
                                max_innovation, max_lateral_delta, time_floor,
                                available, max_tilt, minimum_fraction=0.05,
                                iterations=8):
    """Line-search only the *additional* correction, preserving nominal force."""
    full, saturated = bounded_innovation_force(
        path_force, innovation, time_to_go, weight, max_innovation,
        max_lateral_delta, time_floor, available, max_tilt)
    if not saturated:
        return full, 1.0
    lo, hi = 0.0, 1.0
    best = np.asarray(path_force, dtype=float).copy()
    for _ in range(iterations):
        fraction = (lo + hi) / 2.0
        force, limited = bounded_innovation_force(
            path_force, innovation, time_to_go, weight * fraction,
            max_innovation, max_lateral_delta, time_floor, available, max_tilt)
        if limited:
            hi = fraction
        else:
            lo, best = fraction, force
    if lo < minimum_fraction or np.linalg.norm(best - path_force) <= 0.05:
        return np.asarray(path_force, dtype=float).copy(), 0.0
    return best, lo


class Scenario4BudgetedTerminalAgent(Scenario4GuardedInnovationAgent):
    """Retain the selected route/control and spend only feasible correction."""

    def __init__(self, given_parameters, partial_min_fraction=0.05, **kwargs):
        super().__init__(given_parameters, **kwargs)
        self.partial_min_fraction = float(partial_min_fraction)
        if not np.isfinite(self.partial_min_fraction) or not 0 < self.partial_min_fraction < 1:
            raise ValueError('partial_min_fraction must be in (0,1)')
        self.diagnostics.update(budgeted_partial_steps=0,
                                budgeted_full_steps=0,
                                budgeted_fraction_sum=0.0)

    def _attitude_action(self, axis, jerk, magnitude, gyro, available):
        final_axis = np.asarray(axis, dtype=float)
        final_magnitude = float(magnitude)
        snapshot = self._innovation_snapshot
        target = self.target_index
        if (snapshot is not None and target is not None and self.plan is not None
                and self.plan_duration > 0 and self.filtered_position is not None):
            now, status, states = snapshot
            if (0 <= target < len(status) and status[target] == 1 and
                    np.all(np.isfinite(states[target])) and
                    np.all(np.isfinite(self.filtered_position))):
                time_to_go = float(self.plan_start + self.plan_duration - now)
                weight = smooth_innovation_gate(
                    time_to_go, self.innovation_stop_seconds,
                    self.innovation_start_seconds,
                    self.innovation_fade_seconds, self.innovation_strength)
                if (weight > 0 and np.linalg.norm(
                        states[target, :3] - self.filtered_position) <=
                        self.innovation_max_target_range):
                    elapsed = np.clip(now - self.plan_start, 0, self.plan_duration)
                    reference = sample_curve(self.plan, self.plan_duration, elapsed)[0]
                    tracking_error = float(np.linalg.norm(
                        reference - self.filtered_position))
                    if tracking_error > self.innovation_max_tracking_error:
                        self.diagnostics['guarded_skipped_tracking'] += 1
                    else:
                        endpoint = sample_curve(
                            self.plan, self.plan_duration, self.plan_duration)[0]
                        innovation = observed_endpoint_innovation(
                            states, status, target, time_to_go, endpoint,
                            self.wind_profile, self.profile_blend)
                        if float(np.linalg.norm(innovation)) > 0.05:
                            force, fraction = largest_feasible_innovation(
                                final_axis * final_magnitude, innovation,
                                time_to_go, weight, self.innovation_max_error,
                                self.innovation_max_lateral_delta,
                                self.innovation_time_floor, available,
                                self.max_tilt, self.partial_min_fraction)
                            if fraction == 0.0:
                                self.diagnostics['guarded_skipped_saturated'] += 1
                            else:
                                corrected_magnitude = float(np.linalg.norm(force))
                                if np.all(np.isfinite(force)) and corrected_magnitude > 1e-9:
                                    final_axis = force / corrected_magnitude
                                    final_magnitude = corrected_magnitude
                                    self.diagnostics['guarded_applied_steps'] += 1
                                    self.diagnostics['innovation_applied_steps'] += 1
                                    self.diagnostics['innovation_error_sum'] += float(
                                        np.linalg.norm(innovation))
                                    self.diagnostics['budgeted_fraction_sum'] += fraction
                                    if fraction < 1.0:
                                        self.diagnostics['budgeted_partial_steps'] += 1
                                    else:
                                        self.diagnostics['budgeted_full_steps'] += 1
        self.last_desired_axis = final_axis.copy()
        return Scenario4SubmissionAgent._attitude_action(
            self, final_axis, jerk, final_magnitude, gyro, available)
