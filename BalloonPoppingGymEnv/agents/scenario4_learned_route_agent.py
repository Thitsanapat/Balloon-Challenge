"""Experimental observation-only risk reranking of physically feasible routes.

This is NOT the selected qualification agent. Its frozen weights must come from
audited training fields; the reviewed guarded controller remains the fallback.
No seed, simulator object, hidden balloon trajectory, or outcome is read here.
"""

import json
from pathlib import Path

import numpy as np

from BalloonPoppingGymEnv.agents.submission_scenario4_guarded_innovation_v1 import (
    Scenario4GuardedInnovationAgent,
)


FEATURE_NAMES = (
    'log_time_to_deadline', 'relative_range', 'closing_speed',
    'required_lateral_fraction', 'required_total_fraction',
    'available_acceleration',
)


def capture_features(state, position, velocity, horizon, available):
    """Match the logged first-leg physics features using visible state only."""
    state = np.asarray(state, dtype=float)
    position = np.asarray(position, dtype=float)
    velocity = np.asarray(velocity, dtype=float)
    if (state.shape != (6,) or position.shape != (3,) or velocity.shape != (3,)
            or not np.all(np.isfinite(state)) or
            not np.all(np.isfinite(position)) or
            not np.all(np.isfinite(velocity)) or
            not np.isfinite(horizon) or horizon <= 0 or
            not np.isfinite(available) or available <= 0):
        raise ValueError('Finite visible target and positive horizon/thrust required')
    # routing_features clips the normalized per-node values to [-10, 10].
    relative = np.clip((state[:3] - position) / 150.0, -10.0, 10.0) * 150.0
    relative_v = np.clip((state[3:] - velocity) / 30.0, -10.0, 10.0) * 30.0
    available = max(available, 1e-6)
    required = 2.0 * (relative + relative_v * horizon) / horizon**2
    return np.array((
        np.log1p(horizon), np.linalg.norm(relative) / 150.0,
        -float(np.dot(relative, relative_v)) /
            max(np.linalg.norm(relative), 1e-6) / 30.0,
        np.linalg.norm(required[:2]) / available,
        np.linalg.norm(required + np.array([0., 0., 9.80665])) / available,
        available / 20.0,
    ), dtype=float)


class FrozenCaptureModel:
    def __init__(self, payload):
        if (payload.get('schema') != 'guarded_capture_risk_v1' or
                tuple(payload.get('feature_names', ())) != FEATURE_NAMES):
            raise ValueError('Wrong frozen capture feature schema')
        self.center = np.asarray(payload['center'], dtype=float)
        self.scale = np.asarray(payload['scale'], dtype=float)
        self.coefficients = np.asarray(payload['coefficients'], dtype=float)
        if (self.center.shape != (6,) or self.scale.shape != (6,) or
                self.coefficients.shape != (7,) or
                not np.all(np.isfinite(self.center)) or
                not np.all(np.isfinite(self.scale)) or
                not np.all(np.isfinite(self.coefficients)) or
                np.any(self.scale <= 0)):
            raise ValueError('Invalid frozen capture weights')

    def probability(self, features):
        features = np.asarray(features, dtype=float)
        if features.shape != (6,) or not np.all(np.isfinite(features)):
            raise ValueError('Invalid capture features')
        logit = float(self.coefficients[0] +
                      np.dot((features - self.center) / self.scale,
                             self.coefficients[1:]))
        return float(1.0 / (1.0 + np.exp(-np.clip(logit, -20.0, 20.0))))


class Scenario4LearnedRouteAgent(Scenario4GuardedInnovationAgent):
    """Small learned first-leg preference inside the existing beam-search gate."""

    def __init__(self, given_parameters, model_path, risk_penalty_seconds=1.0,
                 **kwargs):
        super().__init__(given_parameters, **kwargs)
        if not np.isfinite(risk_penalty_seconds) or risk_penalty_seconds < 0:
            raise ValueError('Nonnegative finite route risk penalty required')
        payload = json.loads(Path(model_path).read_text(encoding='utf-8'))
        self.capture_model = FrozenCaptureModel(payload)
        self.risk_penalty_seconds = float(risk_penalty_seconds)
        self._risk_planning_state = None
        self.diagnostics['risk_ranked_nodes'] = 0

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
        features = capture_features(states[index], position, velocity,
                                    horizon, available)
        probability = self.capture_model.probability(features)
        self.diagnostics['risk_ranked_nodes'] += 1
        return original + self.risk_penalty_seconds * (1.0 - probability)
