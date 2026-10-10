"""Scenario-4 route planner with a causal acceleration-residual wind forecast.

The parent chooses launches, routes and actuator commands unchanged.  Only the
position forecast of released balloons is augmented; all measurements come
from the current/past observation stream, with no simulator-state access.
"""

import numpy as np

from BalloonPoppingGymEnv.agents.scenario4_acceleration_forecast import (
    ObservedAccelerationForecast,
)
from BalloonPoppingGymEnv.agents.scenario4_wind_profile_agent import (
    Scenario4WindProfileAgent,
)


class _AugmentedProfile:
    def __init__(self, baseline, observed, blend, active_only=True,
                 max_horizon=3.0):
        self.baseline = baseline
        self.observed = observed
        self.blend = blend
        self.active_only = bool(active_only)
        self.max_horizon = float(max_horizon)
        if not np.isfinite(self.max_horizon) or self.max_horizon <= 0:
            raise ValueError('Positive finite acceleration forecast horizon required')
        self.active_target = None
        self.additional_calls = 0

    def correction(self, states, status, index, horizon):
        base = self.baseline.correction(states, status, index, horizon)
        if (horizon > self.max_horizon or
                (self.active_only and index != self.active_target)):
            return base
        extra = self.observed.correction(
            states, status, index, horizon, self.baseline)
        self.additional_calls += int(bool(np.any(extra)))
        # The parent multiplies our returned offset by profile_blend. Divide
        # here so its normal wind term stays identical and the residual is
        # applied exactly once, as in the paired offline forecast test.
        return base + extra / self.blend


class Scenario4AccelerationWindAgent(Scenario4WindProfileAgent):
    def __init__(self, given_parameters, acceleration_filter_seconds=0.25,
                 acceleration_decay_seconds=1.2,
                 acceleration_strength=0.5,
                 acceleration_max_acceleration=3.0,
                 acceleration_max_residual=1.5,
                 acceleration_max_offset=4.0,
                 acceleration_profile_probe=0.5,
                 acceleration_active_only=True,
                 acceleration_max_horizon=3.0, **kwargs):
        super().__init__(given_parameters, **kwargs)
        if self.profile_blend <= 0:
            raise ValueError('Acceleration augmentation requires positive profile_blend')
        self.acceleration_forecaster = ObservedAccelerationForecast(
            int(given_parameters['balloon']['num']),
            filter_seconds=acceleration_filter_seconds,
            decay_seconds=acceleration_decay_seconds,
            strength=acceleration_strength,
            max_acceleration=acceleration_max_acceleration,
            max_residual_acceleration=acceleration_max_residual,
            max_position_correction=acceleration_max_offset,
            profile_probe_seconds=acceleration_profile_probe)
        self.wind_profile = _AugmentedProfile(
            self.wind_profile, self.acceleration_forecaster,
            self.profile_blend, active_only=acceleration_active_only,
            max_horizon=acceleration_max_horizon)
        self.diagnostics['acceleration_augmented_calls'] = 0

    def _make_plan(self, observation, position, velocity, now):
        # First launch search sees the exact parent's forecast. Once the
        # parent commits to a target, refine that target's forecast only;
        # changing every candidate's coordinates reordered the whole route
        # and regressed badly in the first two official development flights.
        self.wind_profile.active_target = (
            self.route[0] if self.route else None)
        return super()._make_plan(observation, position, velocity, now)

    def get_action(self, observation):
        self.acceleration_forecaster.observe(
            observation['simulation_time'], observation['balloon_states'],
            observation['balloon_status'])
        result = super().get_action(observation)
        self.diagnostics['acceleration_augmented_calls'] = (
            self.wind_profile.additional_calls)
        return result
