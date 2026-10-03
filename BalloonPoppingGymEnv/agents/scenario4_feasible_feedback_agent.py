"""Observation-only tracking feedback constrained by the published thrust envelope.

The established Scenario 4 planner remains responsible for targets and curves.
This variant changes only the frequency of its position/velocity feedback: at
each sensor sample, it chooses the strongest gain whose requested acceleration
fits the current static thrust and tilt limits.  A slow rise avoids amplifying
GNSS noise or issuing abrupt attitude changes after a plan refresh.
"""

import numpy as np

from BalloonPoppingGymEnv.agents.scenario4_wind_profile_agent import (
    Scenario4WindProfileAgent,
)
from BalloonPoppingGymEnv.agents.submission_sqp_v2 import (
    G,
    allocate_acceleration,
    sample_curve,
)


def feasible_feedback_gain(reference_acceleration, position_error, velocity_error,
                           disturbance, available, max_tilt, gains,
                           tolerance=0.1):
    """Choose the largest feasible gain, or the least saturated candidate.

    This uses the same static acceleration allocator as the flight controller.
    The calculation is deliberately independent of the balloon state, seed, or
    simulator internals.  ``gains`` must be positive and sorted ascending.
    """
    gains = np.asarray(gains, dtype=float)
    vectors = [np.asarray(value, dtype=float) for value in (
        reference_acceleration, position_error, velocity_error, disturbance,
    )]
    if (gains.ndim != 1 or not len(gains) or
            np.any(~np.isfinite(gains)) or np.any(gains <= 0) or
            np.any(np.diff(gains) <= 0) or
            any(value.shape != (3,) or not np.all(np.isfinite(value))
                for value in vectors) or
            not np.isfinite(available) or available < 0 or
            not np.isfinite(max_tilt) or not 0 <= max_tilt < np.pi / 2 or
            not np.isfinite(tolerance) or tolerance < 0):
        raise ValueError("Invalid feasible-feedback inputs")
    acceleration, p_error, v_error, disturbance = vectors
    residuals = []
    for gain in gains:
        requested = acceleration + gain**2 * p_error + 2 * gain * v_error
        requested = requested - G - disturbance
        allocated = allocate_acceleration(requested, available, max_tilt)
        residuals.append(float(np.linalg.norm(allocated - requested)))
    residuals = np.asarray(residuals)
    feasible = np.flatnonzero(residuals <= tolerance)
    if len(feasible):
        selected = int(feasible[-1])
        return float(gains[selected]), float(residuals[selected]), True
    # When the reference itself is outside the envelope, prefer the candidate
    # with the smallest force loss.  Equal losses favor stronger tracking.
    selected = int(np.argmin(residuals[::-1]))
    selected = len(gains) - 1 - selected
    return float(gains[selected]), float(residuals[selected]), False


class Scenario4FeasibleFeedbackAgent(Scenario4WindProfileAgent):
    """Keep the live wind-profile planner and adapt only the tracking gain."""

    def __init__(self, given_parameters, feedback_min_frequency=0.7,
                 feedback_max_frequency=1.4, feedback_gain_candidates=9,
                 feedback_rise_rate=1.5, feedback_safety=0.15, **kwargs):
        super().__init__(given_parameters, **kwargs)
        self.feedback_min_frequency = float(feedback_min_frequency)
        self.feedback_max_frequency = float(feedback_max_frequency)
        self.feedback_gain_candidates = int(feedback_gain_candidates)
        self.feedback_rise_rate = float(feedback_rise_rate)
        self.feedback_safety = float(feedback_safety)
        settings = (self.feedback_min_frequency, self.feedback_max_frequency,
                    self.feedback_rise_rate, self.feedback_safety)
        if (not np.all(np.isfinite(settings)) or
                self.feedback_min_frequency <= 0 or
                self.feedback_max_frequency < self.feedback_min_frequency or
                self.feedback_gain_candidates < 2 or
                self.feedback_rise_rate <= 0 or self.feedback_safety < 0):
            raise ValueError("Invalid feasible-feedback settings")
        self.feedback_gains = np.linspace(
            self.feedback_min_frequency, self.feedback_max_frequency,
            self.feedback_gain_candidates,
        )
        self.feedback_time = None
        self.diagnostics.update(
            feedback_updates=0, feedback_limited_steps=0,
            feedback_no_feasible_steps=0, feedback_gain_sum=0.0,
        )

    def get_action(self, observation):
        self.feedback_time = float(observation["simulation_time"])
        return super().get_action(observation)

    def _filter_sensors(self, gyro, position, velocity):
        filtered = super()._filter_sensors(gyro, position, velocity)
        self._update_feedback_gain(self.feedback_time, filtered[1], filtered[2])
        return filtered

    def _select(self, index, curve, duration, now):
        super()._select(index, curve, duration, now)
        if self.filtered_position is not None and self.filtered_velocity is not None:
            self._update_feedback_gain(now, self.filtered_position,
                                       self.filtered_velocity)

    def _update_feedback_gain(self, now, position, velocity):
        if (self.plan is None or now is None or not np.isfinite(now) or
                self.plan_duration <= 0):
            return
        elapsed = np.clip(now - self.plan_start, 0.0, self.plan_duration)
        reference_p, reference_v, reference_a, _ = sample_curve(
            self.plan, self.plan_duration, elapsed,
        )
        available = max(0.0, self.available_acceleration(now) - self.feedback_safety)
        selected, _, feasible = feasible_feedback_gain(
            reference_a, reference_p - position, reference_v - velocity,
            self.disturbance, available, self.max_tilt, self.feedback_gains,
        )
        # Reduce immediately when the old gain exceeds the current envelope;
        # increase slowly so plan refreshes do not cause a control jump.
        self.tracking_frequency = min(
            selected, self.tracking_frequency + self.feedback_rise_rate * self.dt,
        )
        self.diagnostics["feedback_updates"] += 1
        self.diagnostics["feedback_gain_sum"] += self.tracking_frequency
        if self.tracking_frequency < self.feedback_max_frequency - 1e-9:
            self.diagnostics["feedback_limited_steps"] += 1
        if not feasible:
            self.diagnostics["feedback_no_feasible_steps"] += 1
