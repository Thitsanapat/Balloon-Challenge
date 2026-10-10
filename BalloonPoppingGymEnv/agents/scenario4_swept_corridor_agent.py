"""Rank feasible Scenario 4 routes by *time-aligned* near-path balloons.

The parent still generates and validates rocket splines using only public
parameters and current observations.  This model changes beam ordering: a
balloon contributes only when its extrapolated motion comes close to the
rocket's planned path at the same time.  It does not reward a static cluster
that the rocket cannot actually reach.  The smooth score is a search heuristic,
not a calibrated probability or a substitute for the simulator's collision
test.  No seed, environment internals, or prerecorded trajectory is used.
"""

import numpy as np

from BalloonPoppingGymEnv.agents.scenario4_wind_profile_agent import (
    Scenario4WindProfileAgent,
)


def swept_opportunity(curves, durations, states, released, route, radius,
                      samples=9, max_legs=2, width=0.75,
                      time_scale=6.0):
    """Smooth count of distinct unplanned balloons near a time-aligned path.

    Closest approach is computed between linearly interpolated *relative*
    rocket-balloon positions.  This avoids falsely counting a moving balloon
    merely because it visits a rocket waypoint at a different time.  Sampling
    approximates the quintic path, so the value is for ranking only.
    """
    curves = np.asarray(curves, dtype=float)
    durations = np.asarray(durations, dtype=float)
    states = np.asarray(states, dtype=float)
    if (curves.ndim != 3 or curves.shape[1:] != (6, 3) or
            durations.shape != (len(curves),) or states.ndim != 2 or
            states.shape[1] != 6 or samples < 3 or max_legs < 1 or
            radius <= 0 or width <= 0 or time_scale <= 0 or
            not np.all(np.isfinite(durations)) or np.any(durations <= 0)):
        raise ValueError("Invalid swept-corridor inputs")
    route_ids = set(int(i) for i in route)
    ids = np.array([int(i) for i in released if int(i) not in route_ids],
                   dtype=int)
    if not len(ids) or not len(curves):
        return 0.0
    legs = min(len(curves), max_legs)
    s = np.linspace(0.0, 1.0, samples)
    basis = s[:, None] ** np.arange(6)
    path = np.einsum("sk,nkj->nsj", basis, curves[:legs])
    starts = np.r_[0.0, np.cumsum(durations[:legs - 1])]
    times = starts[:, None] + durations[:legs, None] * s
    observed = states[ids]
    relative = (path[:, :, None, :] - observed[None, None, :, :3] -
                times[:, :, None, None] * observed[None, None, :, 3:6])
    delta = np.diff(relative, axis=1)
    fraction = np.clip(
        -np.sum(relative[:, :-1] * delta, axis=-1) /
        np.maximum(np.sum(delta * delta, axis=-1), 1e-12), 0.0, 1.0)
    closest = relative[:, :-1] + fraction[..., None] * delta
    distance = np.linalg.norm(closest, axis=-1)
    near_time = times[:, :-1, None] + fraction * np.diff(times, axis=1)[:, :, None]
    # A balloon can be near multiple legs, but it can only be popped once.
    score = (np.exp(-0.5 * (np.maximum(distance - radius, 0.0) / width) ** 2)
             * np.exp(-near_time / time_scale))
    return float(np.max(score.reshape(-1, len(ids)), axis=0).sum())


class Scenario4SweptCorridorAgent(Scenario4WindProfileAgent):
    """Prefer validated chains that also pass through observed balloon groups."""

    def __init__(self, given_parameters, corridor_weight=2.5,
                 corridor_samples=9, corridor_legs=2, corridor_width=0.75,
                 corridor_time_scale=6.0, corridor_min_opportunity=0.0,
                 corridor_max_bonus=None, **kwargs):
        super().__init__(given_parameters, **kwargs)
        self.corridor_weight = float(corridor_weight)
        self.corridor_samples = int(corridor_samples)
        self.corridor_legs = int(corridor_legs)
        self.corridor_width = float(corridor_width)
        self.corridor_time_scale = float(corridor_time_scale)
        self.corridor_min_opportunity = float(corridor_min_opportunity)
        self.corridor_max_bonus = (None if corridor_max_bonus is None
                                   else float(corridor_max_bonus))
        if (not np.all(np.isfinite((self.corridor_weight,
                self.corridor_width, self.corridor_time_scale,
                self.corridor_min_opportunity))) or
                self.corridor_weight < 0 or self.corridor_width <= 0 or
                self.corridor_time_scale <= 0 or self.corridor_samples < 3 or
                self.corridor_legs < 1 or self.corridor_min_opportunity < 0 or
                (self.corridor_max_bonus is not None and
                 (not np.isfinite(self.corridor_max_bonus) or
                  self.corridor_max_bonus < 0))):
            raise ValueError("Invalid corridor settings")
        self.diagnostics.update(corridor_ranked_nodes=0,
                                corridor_selected_opportunity=[])

    def _node_priority(self, node, states, released, now):
        base = super()._node_priority(node, states, released, now)
        if self.corridor_weight == 0 or not node[0]:
            return base
        opportunity = swept_opportunity(
            node[2], node[1], states, released, node[0], self.radius,
            samples=self.corridor_samples, max_legs=self.corridor_legs,
            width=self.corridor_width,
            time_scale=self.corridor_time_scale)
        self.diagnostics["corridor_ranked_nodes"] += 1
        bonus = self.corridor_weight * max(
            0.0, opportunity - self.corridor_min_opportunity)
        if self.corridor_max_bonus is not None:
            bonus = min(bonus, self.corridor_max_bonus)
        return base - bonus

    def _make_plan(self, observation, position, velocity, now):
        events = len(self.route_events)
        super()._make_plan(observation, position, velocity, now)
        if len(self.route_events) == events or self.plan is None:
            return
        states = np.asarray(observation["balloon_states"], dtype=float)
        released = np.flatnonzero(
            np.asarray(observation["balloon_status"]).reshape(-1) == 1)
        score = swept_opportunity(
            self.plan[None, :, :], np.array([self.plan_duration]),
            states, released, self.route, self.radius,
            samples=self.corridor_samples, max_legs=1,
            width=self.corridor_width,
            time_scale=self.corridor_time_scale)
        self.diagnostics["corridor_selected_opportunity"].append(
            [float(now), float(score)])
