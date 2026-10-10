"""Rank feasible Scenario 4 routes by climb recovery time at their last hit.

The beam still plans with observed balloons and the published thrust/mass model.
Only its ordering changes; the usual trajectory feasibility checks remain in
charge of accepting a route. A zero weight delegates exactly to the incumbent.
"""

import numpy as np

from BalloonPoppingGymEnv.agents.scenario4_wind_profile_agent import (
    Scenario4WindProfileAgent,
)
from BalloonPoppingGymEnv.agents.submission_sqp_v2 import G


class Scenario4VerticalEnergyAgent(Scenario4WindProfileAgent):
    """Trade a little route time for upward velocity after the planned chain."""

    def __init__(self, given_parameters, vertical_energy_weight=0.0, **kwargs):
        weight = float(vertical_energy_weight)
        if not np.isfinite(weight) or weight < 0:
            raise ValueError("vertical_energy_weight must be finite and nonnegative")
        super().__init__(given_parameters, **kwargs)
        self.vertical_energy_weight = weight

    def _node_priority(self, node, states, released, now):
        priority = super()._node_priority(node, states, released, now)
        if self.vertical_energy_weight == 0.0:
            return priority

        route, durations, _, _, exit_velocity = node
        if not route:
            return priority
        observed_climb = float(states[route[-1], 5])
        exit_climb = float(exit_velocity[2])
        if not np.isfinite(observed_climb) or not np.isfinite(exit_climb):
            return priority
        speed_shortfall = max(0.0, observed_climb) - exit_climb
        if speed_shortfall <= 0.0:
            return priority

        # An optimistic full-throttle estimate of time needed to regain the
        # observed target's climb rate after the final intercept. The existing
        # thrust reserve leaves room for drag and tracking error. Capping the
        # estimate keeps beam ordering useful when little thrust remains.
        exit_time = now + sum(durations)
        climb_acceleration = max(
            0.5, self.available_acceleration(exit_time) + G[2] - self.reserve,
        )
        recovery_seconds = min(4.0, speed_shortfall / climb_acceleration)
        return priority + self.vertical_energy_weight * recovery_seconds
