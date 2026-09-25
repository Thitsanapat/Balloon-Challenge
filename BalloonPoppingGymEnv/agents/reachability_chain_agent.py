"""Candidate screening by multi-horizon acceleration reachability.

The constant-acceleration model is only a cheap ranking heuristic. The parent
planner still re-solves joint derivatives and enforces all physical checks.
"""

import numpy as np
from BalloonPoppingGymEnv.agents.final_approach_agent import FinalApproachAgent
from BalloonPoppingGymEnv.agents.submission_sqp_v2 import G


def reachability_order(states, ids, position, velocity, elapsed, remaining,
                       available, max_tilt):
    ids = np.asarray(ids, dtype=int)
    if not len(ids):
        return []
    probes = np.arange(.5, min(10., remaining)+.001, .5)
    if not len(probes):
        return ids.tolist()
    targets = states[ids, None, :3]+states[ids, None, 3:6]*(elapsed+probes)[None, :, None]
    delta = targets-position-velocity*probes[:, None]
    force = 2*delta/probes[None, :, None]**2-G
    magnitude = np.maximum(np.linalg.norm(force, axis=2), 1e-9)
    tilt_excess = np.maximum(0., np.cos(max_tilt)-force[:, :, 2]/magnitude)
    excess = np.maximum(0., magnitude/max(available, .1)-.98)+tilt_excess
    # Feasible arrival time dominates; infeasible candidates are retained as
    # fallbacks because re-solving the entire chain can alter the initial state.
    scores = np.min(probes[None, :]+20.*excess, axis=1)
    return ids[np.argsort(scores, kind='stable')].tolist()


class ReachabilityChainAgent(FinalApproachAgent):
    def __init__(self, given_parameters, nearest_fraction=.5, **kwargs):
        super().__init__(given_parameters, **kwargs)
        self.nearest_fraction = float(nearest_fraction)
        if not np.isfinite(self.nearest_fraction) or not 0 <= self.nearest_fraction <= 1:
            raise ValueError('nearest_fraction must be in [0,1]')

    def _candidate_ids(self, states, released, route, elapsed, position, velocity, now):
        nearest = super()._candidate_ids(states, released, route, elapsed, position, velocity, now)
        if self.nearest_fraction == 1:
            return nearest
        ids = [int(i) for i in released if i not in route and self.failed_until.get(int(i), 0) <= now]
        ranked = reachability_order(states, ids, position, velocity, elapsed,
            self.launch_time+self.burn_time-now-elapsed,
            self.available_acceleration(now+elapsed), self.max_tilt)
        selected = nearest[:int(self.nearest_fraction*self.branch_targets)]
        for index in ranked:
            if index not in selected:
                selected.append(index)
            if len(selected) >= self.branch_targets:
                break
        return selected
