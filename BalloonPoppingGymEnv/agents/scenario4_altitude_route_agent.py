"""Experimental altitude-aware capture reranking; never selected for live.

The parent still creates and physically validates routes and controls the
rocket.  This optional preference uses only current visible balloon states,
the filtered rocket state and frozen, seed-free logistic weights.
"""

import json
from pathlib import Path

import numpy as np

from BalloonPoppingGymEnv.agents.scenario4_learned_route_agent import (
    FEATURE_NAMES, capture_features,
)
from BalloonPoppingGymEnv.agents.submission_scenario4_guarded_innovation_v1 import (
    Scenario4GuardedInnovationAgent,
)


ALTITUDE_FEATURES = FEATURE_NAMES + ('relative_altitude',)


def altitude_capture_features(state, position, velocity, horizon, available):
    base = capture_features(state, position, velocity, horizon, available)
    altitude = float(np.clip(
        (float(state[2]) - float(position[2])) / 150.0, -10.0, 10.0))
    result = np.r_[base, altitude]
    if not np.all(np.isfinite(result)):
        raise ValueError('Non-finite altitude capture features')
    return result


class FrozenAltitudeCaptureModel:
    def __init__(self, payload):
        if (payload.get('schema') != 'altitude_capture_risk_experimental_v1' or
                tuple(payload.get('feature_names', ())) != ALTITUDE_FEATURES):
            raise ValueError('Wrong experimental capture model schema')
        self.center = np.asarray(payload['center'], dtype=float)
        self.scale = np.asarray(payload['scale'], dtype=float)
        self.coefficients = np.asarray(payload['coefficients'], dtype=float)
        if (self.center.shape != (7,) or self.scale.shape != (7,) or
                self.coefficients.shape != (8,) or
                not np.all(np.isfinite(self.center)) or
                not np.all(np.isfinite(self.scale)) or
                not np.all(np.isfinite(self.coefficients)) or
                np.any(self.scale <= 0)):
            raise ValueError('Invalid altitude capture coefficients')

    def probability(self, features):
        x = np.asarray(features, dtype=float)
        if x.shape != (7,) or not np.all(np.isfinite(x)):
            raise ValueError('Invalid altitude capture features')
        logit = self.coefficients[0] + np.dot(
            (x - self.center) / self.scale, self.coefficients[1:])
        return float(1.0 / (1.0 + np.exp(-np.clip(logit, -20.0, 20.0))))


class Scenario4AltitudeRouteAgent(Scenario4GuardedInnovationAgent):
    def __init__(self, given_parameters, model_path,
                 risk_penalty_seconds=1.0, **kwargs):
        super().__init__(given_parameters, **kwargs)
        self.capture_model = FrozenAltitudeCaptureModel(
            json.loads(Path(model_path).read_text(encoding='utf-8')))
        self.risk_penalty_seconds = float(risk_penalty_seconds)
        if (not np.isfinite(self.risk_penalty_seconds) or
                self.risk_penalty_seconds < 0):
            raise ValueError('Invalid route-risk penalty')
        self._risk_planning_state = None
        self.diagnostics['altitude_ranked_nodes'] = 0

    def _make_plan(self, observation, position, velocity, now):
        self._risk_planning_state = (
            np.asarray(position, dtype=float).copy(),
            np.asarray(velocity, dtype=float).copy(), float(now))
        try:
            return super()._make_plan(observation, position, velocity, now)
        finally:
            self._risk_planning_state = None

    def _node_priority(self, node, states, released, now):
        original = super()._node_priority(node, states, released, now)
        if (not self.launched or self.risk_penalty_seconds <= 0 or
                self._risk_planning_state is None or not node[0]):
            return original
        position, velocity, planned_at = self._risk_planning_state
        if abs(planned_at - now) > 1e-6:
            return original
        index, horizon = int(node[0][0]), float(node[1][0])
        available = self.available_acceleration(now)
        if (index not in released or available <= 0 or
                not np.all(np.isfinite(states[index]))):
            return original
        p = self.capture_model.probability(
            altitude_capture_features(
                states[index], position, velocity, horizon, available))
        self.diagnostics['altitude_ranked_nodes'] += 1
        return original + self.risk_penalty_seconds * (1.0 - p)
