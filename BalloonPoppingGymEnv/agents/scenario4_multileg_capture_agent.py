"""Experimental route-level capture-risk ranking with a frozen model.

Every forecast uses only current observed balloon states, current estimated
rocket state, and feasible spline endpoints. No simulator truth or seed enters
the policy. This is not the selected qualification agent.
"""

import numpy as np

from BalloonPoppingGymEnv.agents.scenario4_learned_route_agent import (
    Scenario4LearnedRouteAgent,
    capture_features,
)
from BalloonPoppingGymEnv.agents.submission_scenario4_guarded_innovation_v1 import (
    Scenario4GuardedInnovationAgent,
    sample_curve,
)


def multileg_failure_probability(node, states, position, velocity, now,
                                 acceleration_at, model, max_legs=3):
    """Discounted mean leg failure estimate along a nominal feasible chain.

    The model was fitted to *executed* first-leg outcomes. Reusing it for
    projected later-leg states is a hypothesis requiring simulator validation,
    not a calibrated probability guarantee.
    """
    route, durations, curves = node[:3]
    if not route or len(route) != len(durations) or len(route) != len(curves):
        raise ValueError('Complete feasible chain required')
    if type(max_legs) is not int or max_legs < 1:
        raise ValueError('Positive integer max_legs required')
    states = np.asarray(states, dtype=float)
    position = np.asarray(position, dtype=float)
    velocity = np.asarray(velocity, dtype=float)
    if (states.ndim != 2 or states.shape[1] != 6 or position.shape != (3,) or
            velocity.shape != (3,) or not np.all(np.isfinite(position)) or
            not np.all(np.isfinite(velocity))):
        raise ValueError('Finite visible rocket and balloon states required')
    elapsed = 0.0
    weighted_failure = 0.0
    weight_sum = 0.0
    for leg, (index, duration, curve) in enumerate(zip(route, durations, curves)):
        if leg >= max_legs:
            break
        index, duration = int(index), float(duration)
        if not 0 <= index < len(states) or not np.all(np.isfinite(states[index])):
            raise ValueError('Valid visible target required')
        if leg:
            position, velocity = sample_curve(curve, duration, 0.0)[:2]
        target = states[index].copy()
        target[:3] += target[3:6] * elapsed
        available = float(acceleration_at(now + elapsed))
        features = capture_features(target, position, velocity,
                                    duration, available)
        probability = model.probability(features)
        if not np.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError('Capture model must return a probability')
        weight = 0.5 ** leg
        weighted_failure += weight * (1.0 - probability)
        weight_sum += weight
        elapsed += duration
    return weighted_failure / weight_sum


class Scenario4MultiLegCaptureAgent(Scenario4LearnedRouteAgent):
    """Rank feasible beam routes by discounted first-three-leg capture risk."""

    def __init__(self, given_parameters, max_risk_legs=3, **kwargs):
        super().__init__(given_parameters, **kwargs)
        self.max_risk_legs = int(max_risk_legs)
        if not 1 <= self.max_risk_legs <= 3:
            raise ValueError('max_risk_legs must be between 1 and 3')
        self.diagnostics['multileg_ranked_nodes'] = 0

    def _node_priority(self, node, states, released, now):
        baseline = Scenario4GuardedInnovationAgent._node_priority(
            self, node, states, released, now)
        if (not self.launched or self.risk_penalty_seconds <= 0 or
                self._risk_planning_state is None or not node[0]):
            return baseline
        position, velocity, planned_at = self._risk_planning_state
        if abs(planned_at - now) > 1e-6 or node[0][0] not in released:
            return baseline
        try:
            risk = multileg_failure_probability(
                node, states, position, velocity, now,
                self.available_acceleration, self.capture_model,
                self.max_risk_legs)
        except ValueError:
            return baseline
        self.diagnostics['multileg_ranked_nodes'] += 1
        return baseline + self.risk_penalty_seconds * risk
