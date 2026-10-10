"""Commit to a forecast model once per target to avoid guidance chatter.

The forecast choice is based on only delayed, observed balloon positions. It
cannot change while the same target remains committed, even if rolling error
statistics fluctuate during the intercept.
"""

import numpy as np

from BalloonPoppingGymEnv.agents.scenario4_capture_confidence_agent import (
    Scenario4CaptureConfidenceAgent, _ConfidenceProfile,
)


class _LatchedConfidenceProfile(_ConfidenceProfile):
    def __init__(self, baseline, observed, evidence, blend, max_horizon):
        super().__init__(baseline, observed, evidence, blend, max_horizon)
        self.use_acceleration = False
        self.target_choices = []

    def set_target(self, index, states, status, now):
        if index == self.active_target:
            return
        self.active_target = index
        self.use_acceleration = False
        if index is None:
            return
        index = int(index)
        states = np.asarray(states, dtype=float)
        status = np.asarray(status).reshape(-1)
        if (0 <= index < len(states) and status[index] == 1 and
                np.all(np.isfinite(states[index]))):
            self.use_acceleration = self.evidence.choose_acceleration(
                states[index, 2])
        self.target_choices.append((float(now), index,
                                    bool(self.use_acceleration)))

    def correction(self, states, status, index, horizon):
        base = self.baseline.correction(states, status, index, horizon)
        if (index != self.active_target or horizon > self.max_horizon or
                not self.use_acceleration):
            return base
        self.eligible_calls += 1
        extra = self.observed.correction(
            states, status, index, horizon, self.baseline)
        self.selected_calls += int(bool(np.any(extra)))
        return base + extra / self.blend


class Scenario4LatchedCaptureAgent(Scenario4CaptureConfidenceAgent):
    def __init__(self, given_parameters, **kwargs):
        super().__init__(given_parameters, **kwargs)
        former = self.wind_profile
        self.wind_profile = _LatchedConfidenceProfile(
            former.baseline, former.observed, former.evidence,
            former.blend, former.max_horizon)
        self.diagnostics['capture_target_choices'] = 0
        self.diagnostics['capture_acceleration_targets'] = 0

    def _make_plan(self, observation, position, velocity, now):
        states = observation['balloon_states']
        status = observation['balloon_status']
        current = self.route[0] if self.route else None
        self.wind_profile.set_target(current, states, status, now)
        result = super()._make_plan(observation, position, velocity, now)
        next_target = self.route[0] if self.route else None
        self.wind_profile.set_target(next_target, states, status, now)
        self.diagnostics['capture_target_choices'] = len(
            self.wind_profile.target_choices)
        self.diagnostics['capture_acceleration_targets'] = sum(
            int(choice[2]) for choice in self.wind_profile.target_choices)
        return result
