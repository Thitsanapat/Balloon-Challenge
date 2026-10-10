"""Experimental risk reranker active only with visible acceleration headroom.

This is a separate development policy, not the qualification submission.
It reads the same observation-only first-leg features and frozen weights as
scenario4_learned_route_agent. The incumbent controller remains unchanged.
"""

import numpy as np

from BalloonPoppingGymEnv.agents.scenario4_learned_route_agent import (
    Scenario4LearnedRouteAgent,
    capture_features,
)


class Scenario4HeadroomLearnedRouteAgent(Scenario4LearnedRouteAgent):
    """Do not let a weak risk estimate rerank nearly full-thrust legs."""

    def __init__(self, given_parameters, max_required_total_fraction=0.8,
                 **kwargs):
        super().__init__(given_parameters, **kwargs)
        fraction = float(max_required_total_fraction)
        if not np.isfinite(fraction) or not 0 < fraction < 1:
            raise ValueError('Acceleration-headroom fraction must be in (0, 1)')
        self.max_required_total_fraction = fraction
        self.diagnostics['risk_headroom_skips'] = 0

    def _node_priority(self, node, states, released, now):
        baseline = super(Scenario4LearnedRouteAgent, self)._node_priority(
            node, states, released, now)
        if (not self.launched or self.risk_penalty_seconds <= 0 or
                self._risk_planning_state is None or not node[0]):
            return baseline
        position, velocity, planned_at = self._risk_planning_state
        if abs(planned_at - now) > 1e-6:
            return baseline
        index, horizon = int(node[0][0]), float(node[1][0])
        available = self.available_acceleration(now)
        if (index not in released or available <= 0 or
                not np.all(np.isfinite(states[index]))):
            return baseline
        features = capture_features(states[index], position, velocity,
                                    horizon, available)
        if features[4] > self.max_required_total_fraction:
            self.diagnostics['risk_headroom_skips'] += 1
            return baseline
        return super()._node_priority(node, states, released, now)
