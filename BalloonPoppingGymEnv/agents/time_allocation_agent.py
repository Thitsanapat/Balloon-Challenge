"""Online continuous leg timing with observed-target insertion.

Inspired by polynomial trajectory time allocation, not a reproduction of a
quadrotor paper's full dynamics. Every candidate re-solves the joint arrival
derivatives and is checked against the existing rocket model. No stored field,
seed, simulator object or prerecorded commands are used.
"""

import numpy as np
from scipy.optimize import minimize
from BalloonPoppingGymEnv.agents.final_approach_agent import FinalApproachAgent
from BalloonPoppingGymEnv.agents.chain_beam_agent import free_chain
from BalloonPoppingGymEnv.agents.constrained_chain_agent import repair_margins
from BalloonPoppingGymEnv.agents.opportunistic_chain_agent import closest_approaches


def optimize_times(agent, route, initial_times, states, position, velocity, now,
                   iterations=20, safety=0.):
    """Return dense-checked (durations, curves, end_v, end_a), or None.

    Times are independent decision variables, bounded by the remaining burn.
    Moving target positions are recomputed at every trial's arrival times.
    Inputs and agent state are never modified; solver success alone is not
    accepted as evidence of physical feasibility.
    """
    nominal = np.asarray(initial_times, dtype=float)
    remaining = agent.launch_time+agent.burn_time-now-1e-4
    if (not len(route) or len(route) != len(nominal) or
            not np.all(np.isfinite(nominal)) or np.min(nominal) < .3 or
            sum(nominal) > remaining or not np.all(np.isfinite(states[list(route)]))):
        return None

    def curves(times):
        arrivals = np.cumsum(times)
        centers = states[list(route), :3]+states[list(route), 3:6]*arrivals[:, None]
        return free_chain(position, velocity, agent.acceleration, centers, times,
                          states[route[-1], 3:6], agent.terminal_weight)

    def objective(times):
        return float(sum(times)+.03*np.sum(((times-nominal)/nominal)**2))

    def margins(times, samples):
        return np.r_[remaining-sum(times), repair_margins(
            agent, curves(times), times, now, samples, safety)]

    x = nominal.copy()
    best = None
    # Preserve a feasible nominal candidate in case the numerical solver fails.
    for samples, budget in ((17, iterations), (65, max(6, iterations//2))):
        try:
            candidate = curves(x)
            starts = (now+np.r_[0., np.cumsum(x)[:-1]])[:, None]
            valid, _, vs, acs = agent._valid(candidate, x, starts, samples=65)
            if np.all(valid) and np.min(margins(x, 65)) >= -1e-8:
                if best is None or objective(x) < objective(best[0]):
                    best = x.copy(), candidate, vs, acs
            result = minimize(objective, x, method='SLSQP',
                bounds=[(max(.3, .6*t), min(agent.leg_horizon, 1.5*t)) for t in nominal],
                constraints={'type': 'ineq', 'fun': lambda t: margins(t, samples)},
                options={'maxiter': budget, 'ftol': 1e-5})
            if not np.all(np.isfinite(result.x)):
                break
            x = result.x
        except (ValueError, np.linalg.LinAlgError, FloatingPointError):
            return best
    try:
        candidate = curves(x)
        starts = (now+np.r_[0., np.cumsum(x)[:-1]])[:, None]
        valid, _, vs, acs = agent._valid(candidate, x, starts, samples=65)
        if (np.all(valid) and np.min(margins(x, 65)) >= -1e-8 and
                (best is None or objective(x) < objective(best[0]))):
            best = x.copy(), candidate, vs, acs
    except (ValueError, np.linalg.LinAlgError, FloatingPointError):
        pass
    return best


class TimeAllocationAgent(FinalApproachAgent):
    def __init__(self, given_parameters, timing_candidates=4, timing_iterations=20,
                 timing_safety=0., timing_only_more_hits=True, **kwargs):
        super().__init__(given_parameters, **kwargs)
        self.timing_candidates = int(timing_candidates)
        self.timing_iterations = int(timing_iterations)
        self.timing_safety = float(timing_safety)
        self.timing_only_more_hits = bool(timing_only_more_hits)
        if (self.timing_candidates < 0 or self.timing_iterations < 1 or
                not np.isfinite(self.timing_safety) or self.timing_safety < 0):
            raise ValueError('Invalid time-allocation settings')
        self.diagnostics.update(timing_solves=0, timing_adopted=0,
                                timing_extra_planned_hits=0)

    def _make_plan(self, observation, position, velocity, now):
        events = len(self.route_events)
        super()._make_plan(observation, position, velocity, now)
        if (not self.timing_candidates or not self.route or
                len(self.route_events) == events):
            return
        route = tuple(self.route)
        times = np.diff(np.r_[now, self.deadlines])
        if min(times) < .3:
            return
        states = np.asarray(observation['balloon_states'], dtype=float)
        released = np.flatnonzero(np.asarray(observation['balloon_status']).reshape(-1) == 1)
        if not set(route).issubset(set(released)):
            return

        def solve(ids, ts):
            self.diagnostics['timing_solves'] += 1
            return optimize_times(self, ids, ts, states, position, velocity, now,
                                  self.timing_iterations, self.timing_safety)

        base = solve(route, times)
        if base is None:
            return
        shorter, curves, _, _ = base
        best = (route, base) if not self.timing_only_more_hits and sum(shorter) < sum(times)-.05 else None
        distances, near_times = closest_approaches(curves, shorter, states)
        arrivals = np.cumsum(shorter)
        candidates = [int(i) for i in released if i not in route and
                      self.failed_until.get(int(i), 0) <= now and distances[i] < 25. and
                      near_times[i] >= .3 and np.min(np.abs(arrivals-near_times[i])) >= .3]
        candidates.sort(key=lambda i: distances[i])
        for index in candidates[:self.timing_candidates]:
            slot = int(np.searchsorted(arrivals, near_times[index]))
            proposal = route[:slot]+(index,)+route[slot:]
            ts = np.diff(np.r_[0., np.insert(arrivals, slot, near_times[index])])
            result = solve(proposal, ts)
            if result is not None and (best is None or
                    (-len(proposal), sum(result[0])) < (-len(best[0]), sum(best[1][0]))):
                best = proposal, result
        if best is None:
            return
        ids, (ts, curves, vs, acs) = best
        self.route, self.deadlines = list(ids), list(now+np.cumsum(ts))
        self.ends_v, self.ends_a = list(vs), list(acs)
        self.route_events[-1] = [now, list(ids), float(sum(ts))]
        self._select(ids[0], curves[0], ts[0], now)
        self.diagnostics['timing_adopted'] += 1
        self.diagnostics['timing_extra_planned_hits'] += len(ids)-len(route)
