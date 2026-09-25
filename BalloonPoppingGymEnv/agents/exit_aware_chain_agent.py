"""Rank beam exits by a bounded, observation-only continuation estimate.

The constant-acceleration estimate is ONLY a beam ordering heuristic. It is not
a reachability certificate, extra reward, or a replacement for the unchanged
joint-trajectory physical checks. No stored fields, seeds, or simulator access.
"""

import numpy as np
from BalloonPoppingGymEnv.agents.submission_time_allocation_v1 import ChainSubmissionAgent, G


def continuation_time(states, ids, position, velocity, elapsed, remaining,
                      available, max_tilt, disturbance, horizon=6.):
    """Earliest sampled constant-force continuation, capped if none is found."""
    cap = max(0., min(float(remaining), float(horizon)))
    if not len(ids) or cap < .5:
        return 0.
    probes = np.arange(.5, cap+1e-8, .5)
    targets = states[np.asarray(ids, dtype=int), None, :3]
    targets = targets + states[np.asarray(ids, dtype=int), None, 3:6]*(elapsed+probes)[None, :, None]
    acceleration = 2*(targets-position-velocity*probes[:, None])/probes[None, :, None]**2
    force = acceleration-G-np.asarray(disturbance)
    magnitude = np.linalg.norm(force, axis=2)
    feasible = ((magnitude <= available) & (magnitude >= .5)
                & (force[:, :, 2] >= magnitude*np.cos(max_tilt)))
    return float(np.min(np.where(feasible, probes[None, :], cap)))


class ExitAwareChainAgent(ChainSubmissionAgent):
    def __init__(self, given_parameters, exit_weight=.5, exit_horizon=6., **kwargs):
        super().__init__(given_parameters, **kwargs)
        self.exit_weight, self.exit_horizon = float(exit_weight), float(exit_horizon)
        if (not np.all(np.isfinite([self.exit_weight, self.exit_horizon]))
                or self.exit_weight < 0 or self.exit_horizon < .5):
            raise ValueError('Finite nonnegative weight and horizon >= .5 required')
        self.diagnostics['exit_rank_calls'] = 0

    def _node_priority(self, node, states, released, now):
        if self.exit_weight == 0:
            return super()._node_priority(node, states, released, now)
        route, times, _, position, velocity = node
        elapsed = float(sum(times))
        ids = [int(i) for i in released if i not in route
               and self.failed_until.get(int(i), 0.) <= now]
        self.diagnostics['exit_rank_calls'] += 1
        tail = continuation_time(states, ids, position, velocity, elapsed,
            self.launch_time+self.burn_time-now-elapsed,
            self.available_acceleration(now+elapsed), self.max_tilt,
            self.disturbance, self.exit_horizon)
        return elapsed+self.exit_weight*tail
