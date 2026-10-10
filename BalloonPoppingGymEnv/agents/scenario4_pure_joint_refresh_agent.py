"""Experimental full-route derivative refresh without actuator-preview logic.

When a committed active leg is no longer feasible from the observed rocket
state, re-solve the *same observed route at the same deadlines*.  The
feasibility check is dense and uses only public parameters and current
observations.  This does not certify future capture or access simulator truth.
"""

import numpy as np

from BalloonPoppingGymEnv.agents.chain_beam_agent import free_chain
from BalloonPoppingGymEnv.agents.scenario4_guarded_innovation_agent import (
    Scenario4GuardedInnovationAgent,
)


class Scenario4PureJointRefreshAgent(Scenario4GuardedInnovationAgent):
    def __init__(self, given_parameters, **kwargs):
        super().__init__(given_parameters, **kwargs)
        self.diagnostics.update(pure_joint_attempts=0, pure_joint_accepted=0)

    def _rescue_committed_leg(self, observation, position, velocity, now,
                              duration):
        if not self.route:
            return False
        route = list(self.route)
        times = np.diff(np.r_[now, self.deadlines])
        status = np.asarray(observation['balloon_status']).reshape(-1)
        states = np.asarray(observation['balloon_states'], dtype=float)
        if (len(times) != len(route) or
                not np.all(np.isfinite(times)) or np.any(times <= 0.12) or
                not np.all(status[route] == 1) or
                not np.all(np.isfinite(states[route])) or
                now + sum(times) >= self.launch_time + self.burn_time - 1e-4):
            return False
        self.diagnostics['pure_joint_attempts'] += 1
        arrivals = np.cumsum(times)
        targets = states[route, :3] + states[route, 3:6] * arrivals[:, None]
        try:
            curves = free_chain(position, velocity, self.acceleration, targets,
                                times, states[route[-1], 3:6],
                                self.terminal_weight)
            starts = (now + np.r_[0.0, arrivals[:-1]])[:, None]
            valid, _, ends_v, ends_a = self._valid(
                curves, times, starts, samples=65)
        except (ValueError, np.linalg.LinAlgError, FloatingPointError):
            return False
        if not np.all(valid) or not np.all(np.isfinite(curves)):
            return False
        self.deadlines = list(now + arrivals)
        self.ends_v, self.ends_a = list(ends_v), list(ends_a)
        self._select(route[0], curves[0], float(times[0]), now)
        self.diagnostics['pure_joint_accepted'] += 1
        return True
