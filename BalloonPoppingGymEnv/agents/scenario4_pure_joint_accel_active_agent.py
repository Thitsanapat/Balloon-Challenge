"""Observation-only short-horizon forecast for the committed Scenario 4 target.

The selected pure-joint planner and controller remain unchanged.  A bounded
acceleration residual is added to the existing wind-profile forecast only for
the active target, 1–3 seconds before its planned intercept.  The estimator
uses current and previous balloon observations, never simulator truth or seed.
"""

import numpy as np

from BalloonPoppingGymEnv.agents.scenario4_acceleration_forecast import (
    ObservedAccelerationForecast,
)
from BalloonPoppingGymEnv.agents.scenario4_acceleration_wind_agent import (
    _AugmentedProfile,
)
from BalloonPoppingGymEnv.agents.scenario4_pure_joint_refresh_agent import (
    Scenario4PureJointRefreshAgent,
)


class _ActiveTargetWindowProfile(_AugmentedProfile):
    def __init__(self, baseline, observed, blend, min_horizon=1.0,
                 max_horizon=3.0):
        super().__init__(baseline, observed, blend, active_only=True,
                         max_horizon=max_horizon)
        self.min_horizon = float(min_horizon)
        if (not np.isfinite(self.min_horizon) or
                not 0 < self.min_horizon <= self.max_horizon):
            raise ValueError('Invalid active-target forecast window')

    def correction(self, states, status, index, horizon):
        if horizon < self.min_horizon:
            return self.baseline.correction(states, status, index, horizon)
        return super().correction(states, status, index, horizon)


class Scenario4PureJointAccelerationAgent(Scenario4PureJointRefreshAgent):
    def __init__(self, given_parameters, acceleration_filter_seconds=0.25,
                 acceleration_decay_seconds=1.2,
                 acceleration_strength=0.5,
                 acceleration_max_acceleration=3.0,
                 acceleration_max_residual=1.5,
                 acceleration_max_offset=4.0,
                 acceleration_profile_probe=0.5,
                 acceleration_min_horizon=1.0,
                 acceleration_max_horizon=3.0, **kwargs):
        super().__init__(given_parameters, **kwargs)
        if self.profile_blend <= 0:
            raise ValueError('Acceleration forecast requires positive profile_blend')
        self.acceleration_forecaster = ObservedAccelerationForecast(
            int(given_parameters['balloon']['num']),
            filter_seconds=acceleration_filter_seconds,
            decay_seconds=acceleration_decay_seconds,
            strength=acceleration_strength,
            max_acceleration=acceleration_max_acceleration,
            max_residual_acceleration=acceleration_max_residual,
            max_position_correction=acceleration_max_offset,
            profile_probe_seconds=acceleration_profile_probe)
        self.wind_profile = _ActiveTargetWindowProfile(
            self.wind_profile, self.acceleration_forecaster,
            self.profile_blend, min_horizon=acceleration_min_horizon,
            max_horizon=acceleration_max_horizon)
        self.diagnostics['active_acceleration_calls'] = 0

    def _make_plan(self, observation, position, velocity, now):
        # An empty route has no active target: first-launch route search stays
        # identical to the selected pure-joint policy.
        self.wind_profile.active_target = self.route[0] if self.route else None
        try:
            return super()._make_plan(observation, position, velocity, now)
        finally:
            self.wind_profile.active_target = self.route[0] if self.route else None

    def get_action(self, observation):
        self.acceleration_forecaster.observe(
            observation['simulation_time'], observation['balloon_states'],
            observation['balloon_status'])
        result = super().get_action(observation)
        self.diagnostics['active_acceleration_calls'] = (
            self.wind_profile.additional_calls)
        return result
