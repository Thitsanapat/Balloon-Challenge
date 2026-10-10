"""Experimental learned-route gate using planned actuator headroom.

Only observation-derived candidate splines and public vehicle parameters are
used. This is a development policy, never the default qualification agent.
"""

import numpy as np

from BalloonPoppingGymEnv.agents.scenario4_learned_joint_agent import (
    Scenario4LearnedJointAgent,
)
from BalloonPoppingGymEnv.agents.submission_scenario4_pure_joint_v2 import (
    G,
    Scenario4PureJointRefreshAgent,
    batch_samples,
)


class Scenario4MarginLearnedJointAgent(Scenario4LearnedJointAgent):
    """Let the capture model rerank only trajectories with actuator reserve."""

    def __init__(self, given_parameters, minimum_margin=0.10, **kwargs):
        super().__init__(given_parameters, **kwargs)
        if not np.isfinite(minimum_margin) or not 0 <= minimum_margin < 1:
            raise ValueError('Minimum normalized margin must be in [0,1)')
        self.minimum_margin = float(minimum_margin)
        self.diagnostics['risk_margin_skips'] = 0

    def _first_leg_margin(self, node, now):
        duration = float(node[1][0])
        curves = np.asarray(node[2][:1], dtype=float)
        if duration <= 0 or curves.shape != (1, 6, 3):
            return -np.inf
        _, _, acceleration, jerk = batch_samples(
            curves, np.array([duration]), samples=17)
        force = acceleration - G - self.disturbance
        magnitude = np.linalg.norm(force, axis=2)
        axis = force / np.maximum(magnitude[:, :, None], 1e-9)
        rate = np.linalg.norm(
            jerk - axis * np.sum(axis * jerk, axis=2)[:, :, None],
            axis=2) / np.maximum(magnitude, 1e-9)
        times = now + duration * np.linspace(0.0, 1.0, 17)
        age = np.maximum(times - self.launch_time, 0.0)
        limit = self.thrust / np.maximum(
            self.dry_mass, self.initial_mass - self.mass_flow * age)
        throttle = magnitude[0] / limit
        throttle_slew = np.abs(np.diff(throttle) / np.diff(times))
        margins = np.r_[1.0 - magnitude[0] / limit,
                        1.0 - rate[0] / self.max_axis_rate,
                        1.0 - throttle_slew /
                        self.control['throttle_rate_limit']]
        if not np.all(np.isfinite(margins)):
            return -np.inf
        return float(np.min(margins))

    def _node_priority(self, node, states, released, now):
        baseline = Scenario4PureJointRefreshAgent._node_priority(
            self, node, states, released, now)
        if (not self.launched or self.risk_penalty_seconds <= 0 or
                self._risk_planning_state is None or not node[0]):
            return baseline
        if self._first_leg_margin(node, now) < self.minimum_margin:
            self.diagnostics['risk_margin_skips'] += 1
            return baseline
        return super()._node_priority(node, states, released, now)
