"""Experimental first-leg trackability ranking for Scenario 4.

The parent still constructs and physically validates every route.  Among
same-depth feasible beam nodes, this variant adds a small cost for a first
leg close to the public thrust, tilt, axis-rate, or throttle-rate limits.
This is an observation/public-parameter heuristic, not SCP or MAV dynamics.
"""

import numpy as np

from BalloonPoppingGymEnv.agents.scenario4_guarded_innovation_agent import (
    Scenario4GuardedInnovationAgent,
)
from BalloonPoppingGymEnv.agents.scenario4_wind_profile_agent import (
    Scenario4WindProfileAgent,
)
from BalloonPoppingGymEnv.agents.submission_sqp_v2 import G, sample_curve


def first_leg_tracking_risk(curve, duration, now, agent, samples=9):
    """Return a bounded [0,1] actuator-utilization cost for one feasible leg.

    This is only a route *ranking* proxy.  It cannot predict actual tracking
    error or replace the existing sampled feasibility check.
    """
    curve = np.asarray(curve, dtype=float)
    duration = float(duration)
    now = float(now)
    if (curve.ndim != 2 or curve.shape[1] != 3 or
            not np.all(np.isfinite(curve)) or not np.isfinite(duration) or
            duration <= 0 or not np.isfinite(now) or samples < 3):
        raise ValueError("Finite curve, positive duration, and samples >= 3 required")
    elapsed = np.linspace(0.0, duration, samples)
    _, _, acceleration, jerk = sample_curve(curve, duration, elapsed)
    force = acceleration - G - agent.disturbance
    magnitude = np.maximum(np.linalg.norm(force, axis=1), 1e-9)
    axis = force / magnitude[:, None]
    age = np.maximum(now + elapsed - agent.launch_time, 0.0)
    mass = np.maximum(agent.dry_mass, agent.initial_mass - agent.mass_flow * age)
    available = agent.thrust / mass
    thrust_use = magnitude / np.maximum(available, 1e-9)
    tilt_use = np.arccos(np.clip(axis[:, 2], -1.0, 1.0)) / agent.max_tilt
    lateral_jerk = jerk - axis * np.sum(axis * jerk, axis=1)[:, None]
    axis_rate = np.linalg.norm(lateral_jerk, axis=1) / magnitude
    axis_rate_use = axis_rate / agent.max_axis_rate
    throttle = magnitude / np.maximum(available, 1e-9)
    throttle_rate = np.abs(np.diff(throttle)) / np.diff(elapsed)
    throttle_rate_use = throttle_rate / agent.control["throttle_rate_limit"]
    utilization = np.maximum.reduce((thrust_use, tilt_use, axis_rate_use))
    # The initial acceleration is inherited from the measured state, so it
    # gets less weight than the portion that this route can actually shape.
    weights = np.linspace(0.2, 1.0, samples)
    near_limit = np.clip((utilization - 0.7) / 0.3, 0.0, 1.0) ** 2
    slew_penalty = np.clip((throttle_rate_use - 0.7) / 0.3, 0.0, 1.0) ** 2
    risk = (np.dot(weights, near_limit) +
            np.dot(weights[1:], slew_penalty)) / (sum(weights) + sum(weights[1:]))
    return float(np.clip(risk, 0.0, 1.0))


class Scenario4TrackabilityAgent(Scenario4GuardedInnovationAgent):
    def __init__(self, given_parameters, trackability_time_weight=1.0,
                 trackability_samples=9, **kwargs):
        super().__init__(given_parameters, **kwargs)
        self.trackability_time_weight = float(trackability_time_weight)
        self.trackability_samples = int(trackability_samples)
        if (not np.isfinite(self.trackability_time_weight) or
                self.trackability_time_weight < 0 or
                self.trackability_samples < 3):
            raise ValueError("Invalid trackability ranking parameters")
        self.diagnostics.update(trackability_ranked_nodes=0,
                                trackability_risk_sum=0.0,
                                trackability_selected_risk=[])

    def _node_priority(self, node, states, released, now):
        base = Scenario4WindProfileAgent._node_priority(
            self, node, states, released, now)
        if not node[0] or self.trackability_time_weight == 0:
            return base
        risk = first_leg_tracking_risk(
            node[2][0], node[1][0], now, self,
            self.trackability_samples)
        self.diagnostics["trackability_ranked_nodes"] += 1
        self.diagnostics["trackability_risk_sum"] += risk
        return base + self.trackability_time_weight * risk

    def _select(self, index, curve, duration, now):
        if self.trackability_time_weight:
            risk = first_leg_tracking_risk(
                curve, duration, now, self, self.trackability_samples)
            self.diagnostics["trackability_selected_risk"].append(
                [float(now), int(index), risk])
        return super()._select(index, curve, duration, now)
