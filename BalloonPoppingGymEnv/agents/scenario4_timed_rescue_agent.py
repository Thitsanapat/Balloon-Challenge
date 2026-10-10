"""Try a small later intercept after same-deadline joint repair is infeasible.

This keeps the observed target order. Every candidate is rebuilt from current
observations and checked against the normal thrust, tilt and rate limits.
No seed, hidden environment state or saved trajectory is used at runtime.
"""

import numpy as np

from BalloonPoppingGymEnv.agents.chain_beam_agent import free_chain
from BalloonPoppingGymEnv.agents.scenario4_pure_joint_refresh_agent import (
    Scenario4PureJointRefreshAgent,
)


class Scenario4TimedRescueAgent(Scenario4PureJointRefreshAgent):
    def __init__(self, given_parameters, timed_rescue_delays=(0.4, 0.8, 1.2),
                 **kwargs):
        super().__init__(given_parameters, **kwargs)
        self.timed_rescue_delays = tuple(float(delay) for delay in timed_rescue_delays)
        if (not 0 < len(self.timed_rescue_delays) <= 4 or
                not np.all(np.isfinite(self.timed_rescue_delays)) or
                any(delay <= 0 or delay > 2 for delay in self.timed_rescue_delays) or
                tuple(sorted(set(self.timed_rescue_delays))) != self.timed_rescue_delays):
            raise ValueError('Positive, increasing intercept delays of at most 2 seconds required')
        self.diagnostics.update(timed_rescue_attempts=0, timed_rescue_accepted=0,
                                timed_rescue_delay_sum=0.0)

    def _rescue_committed_leg(self, observation, position, velocity, now, duration):
        if super()._rescue_committed_leg(observation, position, velocity, now,
                                         duration):
            return True
        route = list(self.route)
        if not route:
            return False
        times = np.diff(np.r_[now, self.deadlines])
        status = np.asarray(observation['balloon_status']).reshape(-1)
        states = np.asarray(observation['balloon_states'], dtype=float)
        if (len(times) != len(route) or
                not np.all(np.isfinite(times)) or np.any(times <= 0.12) or
                not np.all(status[route] == 1) or
                not np.all(np.isfinite(states[route]))):
            return False
        cutoff = self.launch_time + self.burn_time - 1e-4
        for delay in self.timed_rescue_delays:
            adjusted = times.copy()
            adjusted[0] += delay
            arrivals = np.cumsum(adjusted)
            if now + arrivals[-1] >= cutoff:
                break
            self.diagnostics['timed_rescue_attempts'] += 1
            targets = states[route, :3] + states[route, 3:6] * arrivals[:, None]
            try:
                curves = free_chain(position, velocity, self.acceleration,
                                    targets, adjusted, states[route[-1], 3:6],
                                    self.terminal_weight)
                starts = (now + np.r_[0.0, arrivals[:-1]])[:, None]
                valid, _, ends_v, ends_a = self._valid(
                    curves, adjusted, starts, samples=65)
            except (ValueError, np.linalg.LinAlgError, FloatingPointError):
                continue
            if not np.all(valid) or not np.all(np.isfinite(curves)):
                continue
            self.deadlines = list(now + arrivals)
            self.ends_v, self.ends_a = list(ends_v), list(ends_a)
            self._select(route[0], curves[0], float(adjusted[0]), now)
            self.diagnostics['timed_rescue_accepted'] += 1
            self.diagnostics['timed_rescue_delay_sum'] += delay
            return True
        return False
