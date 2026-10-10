"""Re-solve arrival derivatives after an infeasible committed-leg refresh.

Retains observed targets and deadlines, but releases stale arrival derivatives.
Only this agent changes; official physics and observations are untouched.
"""
import numpy as np
from BalloonPoppingGymEnv.agents.chain_beam_agent import free_chain
from BalloonPoppingGymEnv.agents.scenario4_wind_profile_agent import Scenario4WindProfileAgent


class Scenario4JointRefreshAgent(Scenario4WindProfileAgent):
    def __init__(self, given_parameters, joint_refresh_min_duration=0.3,
                 joint_refresh_allow_shortening=True, **kwargs):
        super().__init__(given_parameters, **kwargs)
        self.joint_refresh_min_duration = float(joint_refresh_min_duration)
        self.joint_refresh_allow_shortening = bool(joint_refresh_allow_shortening)
        if not np.isfinite(self.joint_refresh_min_duration) or self.joint_refresh_min_duration < .12:
            raise ValueError("Joint refresh duration must be finite and at least .12 s")
        self.diagnostics.update(joint_refresh_attempts=0, joint_refresh_accepted=0,
                                joint_refresh_dropped_legs=0)

    def _rescue_committed_leg(self, observation, position, velocity, now, duration):
        if not self.route or duration < self.joint_refresh_min_duration:
            return False
        states = np.asarray(observation['balloon_states'], dtype=float)
        status = np.asarray(observation['balloon_status']).reshape(-1)
        times = np.diff(np.r_[now, self.deadlines])
        if (len(times) != len(self.route) or not np.all(np.isfinite(times))
                or np.any(times < self.joint_refresh_min_duration)
                or self.deadlines[-1] >= self.launch_time + self.burn_time):
            return False
        sizes = [len(self.route)]
        if self.joint_refresh_allow_shortening:
            sizes += [min(2, len(self.route)), 1]
        for count in dict.fromkeys(sizes):
            route = self.route[:count]
            if any(status[i] != 1 for i in route) or not np.all(np.isfinite(states[route])):
                continue
            ts = times[:count]
            targets = states[route, :3] + states[route, 3:] * np.cumsum(ts)[:, None]
            self.diagnostics['joint_refresh_attempts'] += 1
            try:
                curves = free_chain(position, velocity, self.acceleration, targets,
                                    ts, states[route[-1], 3:], self.terminal_weight)
                starts = (now + np.r_[0., np.cumsum(ts)[:-1]])[:, None]
                valid, _, end_v, end_a = self._valid(curves, ts, starts, samples=65)
            except (ValueError, np.linalg.LinAlgError, FloatingPointError):
                continue
            if not np.all(np.isfinite(curves)) or not np.all(valid):
                continue
            self.diagnostics['joint_refresh_dropped_legs'] += len(self.route) - count
            self.route = list(route)
            self.deadlines = list(now + np.cumsum(ts))
            self.ends_v, self.ends_a = list(end_v), list(end_a)
            self._select(route[0], curves[0], float(ts[0]), now)
            self.diagnostics['joint_refresh_accepted'] += 1
            return True
        return False
