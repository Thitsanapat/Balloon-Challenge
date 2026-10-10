"""Causal forecast selection for the committed Scenario-4 intercept.

The selector compares predictions made two seconds ago with *later observed*
balloon positions.  It never reads environment info, the field seed, or future
states.  A radial Gaussian capture proxy ranks the two forecasts after adding
the same tracking-error floor to each; it is not a calibrated hit probability.
"""

from collections import deque

import numpy as np

from BalloonPoppingGymEnv.agents.scenario4_acceleration_forecast import (
    ObservedAccelerationForecast,
)
from BalloonPoppingGymEnv.agents.scenario4_wind_profile_agent import (
    Scenario4WindProfileAgent,
)


class CaptureEvidence:
    """Evaluate paired, timestamped forecasts using only later observations."""

    def __init__(self, horizon=2.0, sample_period=0.5, peers=8,
                 history=80, altitude_width=25.0, min_evidence=12,
                 tracking_error=1.5, capture_radius=1.5,
                 minimum_gain=0.015):
        values = (horizon, sample_period, altitude_width, tracking_error,
                  capture_radius, minimum_gain)
        if (not np.all(np.isfinite(values)) or horizon <= 0 or
                sample_period <= 0 or altitude_width <= 0 or
                tracking_error < 0 or capture_radius <= 0 or
                minimum_gain < 0 or peers < 1 or history < 1 or
                min_evidence < 1):
            raise ValueError('Invalid capture evidence settings')
        self.horizon = float(horizon)
        self.sample_period = float(sample_period)
        self.peers = int(peers)
        self.altitude_width = float(altitude_width)
        self.min_evidence = int(min_evidence)
        self.tracking_error = float(tracking_error)
        self.capture_radius = float(capture_radius)
        self.minimum_gain = float(minimum_gain)
        self.pending = deque()
        self.history = deque(maxlen=int(history))
        self.next_sample = -np.inf

    def observe(self, now, states, status, profile, forecaster, blend,
                active_target=None, reference_altitude=0.0):
        states = np.asarray(states, dtype=float)
        status = np.asarray(status).reshape(-1)
        now = float(now)
        if (states.ndim != 2 or states.shape[1] != 6 or
                len(status) != len(states) or not np.isfinite(now)):
            raise ValueError('Invalid capture evidence observation')
        while self.pending and self.pending[0][0] <= now + 1e-8:
            _, index, altitude, baseline, alternative = self.pending.popleft()
            if status[index] != 1 or not np.all(np.isfinite(states[index])):
                continue
            actual = states[index, :2]
            self.history.append((altitude,
                                 float(np.sum((baseline - actual)**2)),
                                 float(np.sum((alternative - actual)**2))))
        if now + 1e-8 < self.next_sample:
            return
        self.next_sample = now + self.sample_period
        valid = np.flatnonzero((status == 1) & np.all(np.isfinite(states), axis=1))
        if not len(valid):
            return
        if active_target is not None and int(active_target) in valid:
            reference_altitude = states[int(active_target), 2]
        ids = valid[np.argsort(np.abs(states[valid, 2] - reference_altitude))
                    [:self.peers]]
        for index in ids:
            base = profile.correction(states, status, int(index), self.horizon)
            extra = forecaster.correction(
                states, status, int(index), self.horizon, profile)
            linear = states[index, :2] + self.horizon * states[index, 3:5]
            predicted = linear + blend * base
            self.pending.append((now + self.horizon, int(index),
                                 float(states[index, 2]), predicted,
                                 predicted + extra))

    def capture_gain(self, altitude):
        local = [(old, new) for z, old, new in self.history
                 if abs(z - altitude) <= self.altitude_width]
        if len(local) < self.min_evidence:
            return None
        mse = np.mean(np.asarray(local), axis=0)
        variances = mse + self.tracking_error**2
        probabilities = -np.expm1(-self.capture_radius**2 /
                                   np.maximum(variances, 1e-12))
        return float(probabilities[1] - probabilities[0])

    def choose_acceleration(self, altitude):
        gain = self.capture_gain(float(altitude))
        return gain is not None and gain >= self.minimum_gain


class _ConfidenceProfile:
    def __init__(self, baseline, observed, evidence, blend, max_horizon):
        self.baseline = baseline
        self.observed = observed
        self.evidence = evidence
        self.blend = blend
        self.max_horizon = max_horizon
        self.active_target = None
        self.eligible_calls = 0
        self.selected_calls = 0

    def correction(self, states, status, index, horizon):
        base = self.baseline.correction(states, status, index, horizon)
        if (index != self.active_target or horizon > self.max_horizon or
                not self.evidence.choose_acceleration(states[index, 2])):
            return base
        self.eligible_calls += 1
        extra = self.observed.correction(
            states, status, index, horizon, self.baseline)
        self.selected_calls += int(bool(np.any(extra)))
        return base + extra / self.blend


class Scenario4CaptureConfidenceAgent(Scenario4WindProfileAgent):
    """Keep the incumbent route; switch its active forecast only with evidence."""

    def __init__(self, given_parameters, acceleration_filter_seconds=0.25,
                 acceleration_decay_seconds=1.2,
                 acceleration_strength=0.5,
                 acceleration_max_acceleration=3.0,
                 acceleration_max_residual=1.5,
                 acceleration_max_offset=4.0,
                 acceleration_profile_probe=0.5,
                 acceleration_max_horizon=3.0,
                 evidence_horizon=2.0, evidence_sample_period=0.5,
                 evidence_peers=8, evidence_history=80,
                 evidence_altitude_width=25.0,
                 evidence_min_count=12,
                 evidence_tracking_error=1.5,
                 evidence_minimum_gain=0.015, **kwargs):
        super().__init__(given_parameters, **kwargs)
        if self.profile_blend <= 0 or acceleration_max_horizon <= 0:
            raise ValueError('Positive blend and forecast horizon required')
        self.acceleration_forecaster = ObservedAccelerationForecast(
            int(given_parameters['balloon']['num']),
            filter_seconds=acceleration_filter_seconds,
            decay_seconds=acceleration_decay_seconds,
            strength=acceleration_strength,
            max_acceleration=acceleration_max_acceleration,
            max_residual_acceleration=acceleration_max_residual,
            max_position_correction=acceleration_max_offset,
            profile_probe_seconds=acceleration_profile_probe)
        self.capture_evidence = CaptureEvidence(
            horizon=evidence_horizon, sample_period=evidence_sample_period,
            peers=evidence_peers, history=evidence_history,
            altitude_width=evidence_altitude_width,
            min_evidence=evidence_min_count,
            tracking_error=evidence_tracking_error,
            capture_radius=float(given_parameters['balloon']['radius']),
            minimum_gain=evidence_minimum_gain)
        self.wind_profile = _ConfidenceProfile(
            self.wind_profile, self.acceleration_forecaster,
            self.capture_evidence, self.profile_blend,
            float(acceleration_max_horizon))
        self.diagnostics.update(capture_evidence_count=0,
                                capture_eligible_calls=0,
                                capture_selected_calls=0)

    def _make_plan(self, observation, position, velocity, now):
        self.wind_profile.active_target = self.route[0] if self.route else None
        return super()._make_plan(observation, position, velocity, now)

    def get_action(self, observation):
        now = float(observation['simulation_time'])
        states = observation['balloon_states']
        status = observation['balloon_status']
        self.acceleration_forecaster.observe(now, states, status)
        active = self.route[0] if self.route else None
        altitude = (self.filtered_position[2] if self.filtered_position is not None
                    else self.elevation)
        self.capture_evidence.observe(
            now, states, status, self.wind_profile.baseline,
            self.acceleration_forecaster, self.profile_blend,
            active_target=active, reference_altitude=altitude)
        action = super().get_action(observation)
        self.diagnostics['capture_evidence_count'] = len(self.capture_evidence.history)
        self.diagnostics['capture_eligible_calls'] = self.wind_profile.eligible_calls
        self.diagnostics['capture_selected_calls'] = self.wind_profile.selected_calls
        return action
