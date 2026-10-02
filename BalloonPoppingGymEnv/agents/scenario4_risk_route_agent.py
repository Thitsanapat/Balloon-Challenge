"""Observation-only risk ranking for Scenario 4's joint route search.

The existing chain planner certifies nominal trajectory feasibility.  This
variant changes *only* beam ordering and the maximum lookahead depth.  Its
preference for early, low-effort intercepts is a heuristic for forecast error
and unmodeled gusts, not a guarantee of collision probability.  It never uses
the simulator wind field, seed, ``info``, or balloon future states.
"""

import numpy as np

from BalloonPoppingGymEnv.agents.scenario4_wind_profile_agent import (
    Scenario4WindProfileAgent,
)
from BalloonPoppingGymEnv.agents.submission_sqp_v2 import G


class Scenario4RiskRouteAgent(Scenario4WindProfileAgent):
    """Retain physically feasible beam nodes with lower short-term risk.

    ``risk_depth_limit`` limits speculative multi-hit lookahead, not the
    number of balloons the agent can pop over the flight: the policy replans
    from every new observation.  A value of zero leaves parent depth intact.
    All weights are expressed as seconds added to the original duration key.
    """

    def __init__(self, given_parameters, risk_depth_limit=5,
                 risk_first_horizon=4.0, risk_future_horizon=5.0,
                 risk_first_weight=0.20, risk_future_weight=0.035,
                 risk_drift_weight=0.01, risk_force_soft_limit=0.80,
                 risk_force_weight=1.5, **kwargs):
        super().__init__(given_parameters, **kwargs)
        self.risk_depth_limit = int(risk_depth_limit)
        self.risk_first_horizon = float(risk_first_horizon)
        self.risk_future_horizon = float(risk_future_horizon)
        self.risk_first_weight = float(risk_first_weight)
        self.risk_future_weight = float(risk_future_weight)
        self.risk_drift_weight = float(risk_drift_weight)
        self.risk_force_soft_limit = float(risk_force_soft_limit)
        self.risk_force_weight = float(risk_force_weight)
        values = (self.risk_first_horizon, self.risk_future_horizon,
                  self.risk_first_weight, self.risk_future_weight,
                  self.risk_drift_weight, self.risk_force_soft_limit,
                  self.risk_force_weight)
        if (self.risk_depth_limit < 0 or not np.all(np.isfinite(values)) or
                min(values) < 0 or not 0 < self.risk_force_soft_limit < 1):
            raise ValueError("Invalid risk-route settings")
        self.diagnostics.update(risk_ranked_nodes=0, risk_depth_capped_plans=0)

    def _node_priority(self, node, states, released, now):
        base = float(super()._node_priority(node, states, released, now))
        ids, times, curves = node[:3]
        if not ids:
            return base
        durations = np.asarray(times, dtype=float)
        arrival = np.cumsum(durations)
        first_late = max(0., arrival[0] - self.risk_first_horizon)
        future_late = np.maximum(0., arrival - self.risk_future_horizon)
        penalty = (self.risk_first_weight * first_late**2 +
                   self.risk_future_weight * float(np.mean(future_late**2)))

        # The observed target drift, not a hidden atmospheric model, indicates
        # how much position error a long extrapolation could accumulate.
        if self.risk_drift_weight:
            horizontal_speed = np.linalg.norm(states[list(ids), 3:5], axis=1)
            penalty += self.risk_drift_weight * float(np.mean(arrival * horizontal_speed))

        # A feasible curve can still run close to available thrust.  Compare
        # the required thrust at each leg midpoint with the *public* rocket
        # motor/mass model; a gust then has less margin before saturation.
        if self.risk_force_weight:
            c = np.asarray(curves, dtype=float)
            t2 = durations[:, None]**2
            midpoint_accel = (2*c[:, 2] + 3*c[:, 3] +
                              3*c[:, 4] + 2.5*c[:, 5]) / t2
            required = np.linalg.norm(midpoint_accel - G - self.disturbance, axis=1)
            midpoints = now + arrival - 0.5*durations
            available = np.asarray([self.available_acceleration(t)
                                    for t in midpoints])
            utilization = required / np.maximum(available, 1e-9)
            excess = np.maximum(0., utilization - self.risk_force_soft_limit)
            penalty += self.risk_force_weight * float(np.mean(
                (excess / (1. - self.risk_force_soft_limit))**2))
        self.diagnostics['risk_ranked_nodes'] += 1
        return base + penalty

    def _make_plan(self, observation, position, velocity, now):
        original_depth = self.search_depth
        if self.risk_depth_limit and original_depth > self.risk_depth_limit:
            self.search_depth = self.risk_depth_limit
            self.diagnostics['risk_depth_capped_plans'] += 1
        try:
            return super()._make_plan(observation, position, velocity, now)
        finally:
            self.search_depth = original_depth
