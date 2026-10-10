"""Safety-gated active-leg forecast correction for Scenario 4.

The route and spline planner are unchanged. Corrections to the currently
observed target are suppressed if the desired extra force cannot be allocated
within actuator limits or if spline tracking is already poor. No simulator
truth, random seed, or future balloon trajectory is available to this agent.
"""

import numpy as np

from BalloonPoppingGymEnv.agents.scenario4_forecast_innovation_agent import (
    Scenario4ForecastInnovationAgent,
    bounded_innovation_force,
    observed_endpoint_innovation,
    smooth_innovation_gate,
)
from BalloonPoppingGymEnv.agents.scenario4_wind_profile_agent import (
    Scenario4WindProfileAgent,
)
from BalloonPoppingGymEnv.agents.submission_sqp_v2 import sample_curve


def accept_terminal_correction(tracking_error, max_tracking_error, saturated):
    """A failed force allocation must never replace feasible nominal control."""
    if (not np.isfinite(tracking_error) or
            not np.isfinite(max_tracking_error) or max_tracking_error <= 0):
        raise ValueError("Finite nonnegative tracking bounds required")
    return not saturated and tracking_error <= max_tracking_error


class Scenario4GuardedInnovationAgent(Scenario4ForecastInnovationAgent):
    def __init__(self, given_parameters, innovation_tracking_radius_multiple=2.0,
                 **kwargs):
        super().__init__(given_parameters, **kwargs)
        self.innovation_max_tracking_error = (
            float(innovation_tracking_radius_multiple)
            * float(given_parameters["balloon"]["radius"]))
        if (not np.isfinite(self.innovation_max_tracking_error) or
                self.innovation_max_tracking_error <= 0):
            raise ValueError("Positive tracking gate required")
        self.diagnostics.update(guarded_skipped_tracking=0,
                                guarded_skipped_saturated=0,
                                guarded_applied_steps=0)

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
                            force, saturated = bounded_innovation_force(
                                final_axis * final_magnitude, innovation,
                                time_to_go, weight,
                                self.innovation_max_error,
                                self.innovation_max_lateral_delta,
                                self.innovation_time_floor, available,
                                self.max_tilt)
                            if saturated:
                                self.diagnostics["guarded_skipped_saturated"] += 1
                            elif accept_terminal_correction(
                                    tracking_error,
                                    self.innovation_max_tracking_error,
                                    saturated):
                                corrected_magnitude = float(np.linalg.norm(force))
                                if (np.all(np.isfinite(force)) and
                                        corrected_magnitude > 1e-9):
                                    final_axis = force / corrected_magnitude
                                    final_magnitude = corrected_magnitude
                                    self.diagnostics["guarded_applied_steps"] += 1
                                    self.diagnostics["innovation_applied_steps"] += 1
                                    self.diagnostics["innovation_error_sum"] += float(
                                        np.linalg.norm(innovation))
        self.last_desired_axis = final_axis.copy()
        # Bypass the unguarded parent's override; use the reviewed attitude
        # controller with the resulting (or unchanged nominal) force.
        return Scenario4WindProfileAgent._attitude_action(
            self, final_axis, jerk, final_magnitude, gyro, available)
