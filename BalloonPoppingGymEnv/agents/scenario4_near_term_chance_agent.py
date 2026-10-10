"""Near-term-only chance ranking; long-horizon route order stays unchanged.

The causal forecast uncertainty model was checked only over 1–3 seconds.
Applying its extrapolation to an entire chain lost a development-seed pop.
This conservative variant uses it only for the first leg within that window.
"""

import numpy as np

from BalloonPoppingGymEnv.agents.scenario4_chance_route_agent import (
    Scenario4ChanceRouteAgent,
    chance_route_score,
)
from BalloonPoppingGymEnv.agents.scenario4_wind_profile_agent import (
    Scenario4WindProfileAgent,
)


class Scenario4NearTermChanceAgent(Scenario4ChanceRouteAgent):
    def __init__(self, given_parameters, chance_max_first_horizon=3.0,
                 **kwargs):
        super().__init__(given_parameters, **kwargs)
        self.chance_max_first_horizon = float(chance_max_first_horizon)
        if (not np.isfinite(self.chance_max_first_horizon) or
                self.chance_max_first_horizon <= 0):
            raise ValueError("Positive near-term horizon required")
        self.diagnostics["chance_near_term_ranked_nodes"] = 0

    def _node_priority(self, node, states, released, now):
        # Long-chain candidates retain precisely the reviewed ordering.
        base = Scenario4WindProfileAgent._node_priority(
            self, node, states, released, now)
        if (not node[0] or node[1][0] > self.chance_max_first_horizon or
                (self.chance_route_weight == 0 and
                 self.chance_incidental_weight == 0)):
            return base
        planned, incidental = chance_route_score(
            node[2][:1], node[1][:1], states, released, node[0][:1],
            self.radius, self.chance_history, self.chance_navigation_sigma,
            self.chance_samples, 1, self.chance_time_scale)
        self.diagnostics["chance_near_term_ranked_nodes"] += 1
        return (base + self.chance_route_weight * (1 - planned)
                - self.chance_incidental_weight * incidental)
