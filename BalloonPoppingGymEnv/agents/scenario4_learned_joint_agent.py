"""Experimental learned route ranking on the selected pure-joint controller.

The flight controller is the reviewed standalone pure-joint source. The only
new behavior is observation-only reranking of physical beam-search proposals.
This is NOT the live qualification agent.
"""

import json
from pathlib import Path

import numpy as np

from BalloonPoppingGymEnv.agents.scenario4_learned_route_agent import (
    FrozenCaptureModel,
    capture_features,
)
from BalloonPoppingGymEnv.agents.submission_scenario4_pure_joint_v2 import (
    Scenario4PureJointRefreshAgent,
)


class Scenario4LearnedJointAgent(Scenario4PureJointRefreshAgent):
    """Add frozen first-leg risk preference without replacing joint repair."""

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
