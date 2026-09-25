"""Observation-only extra-hit planning with constrained arrival derivatives.

SLSQP adjusts waypoint velocities and accelerations, preserving positions,
arrival times and C2 continuity exactly. Full sampled physical checks are
constraints of the solve, not just a penalty in the trajectory objective.
An unsuccessful repair leaves the original feasible route untouched.
"""

import numpy as np
from scipy.optimize import minimize
from BalloonPoppingGymEnv.agents.chain_launch_agent import ChainLaunchAgent
from BalloonPoppingGymEnv.agents.chain_beam_agent import free_chain, free_chain_matrices
from BalloonPoppingGymEnv.agents.opportunistic_chain_agent import closest_approaches
from BalloonPoppingGymEnv.agents.submission_sqp_v2 import batch_samples, G


def constraint_margins(agent, curves, durations, now, samples=25):
    """Nonnegative means feasible in the same model as MomentumBeamAgent."""
    durations = np.asarray(durations)
    p, _, a, j = batch_samples(curves, durations, samples)
    times = now+np.r_[0., np.cumsum(durations)[:-1]][:, None]+durations[:, None]*np.linspace(0, 1, samples)
    limit = agent.thrust/np.maximum(agent.dry_mass, agent.initial_mass-
                                  agent.mass_flow*np.maximum(times-agent.launch_time, 0.))
    force = a-G-agent.disturbance
    magnitude = np.maximum(np.linalg.norm(force, axis=2), 1e-9)
    axis = force/magnitude[:, :, None]
    transverse = j-axis*np.sum(axis*j, axis=2)[:, :, None]
    rate = np.linalg.norm(transverse, axis=2)/magnitude
    throttle = magnitude/limit
    slew = np.diff(throttle, axis=1)/np.diff(times, axis=1)
    reserve = agent.reserve*np.linspace(0., 1., samples)
    return np.r_[((limit-reserve-magnitude)/12.).ravel(), ((magnitude-.5)/12.).ravel(),
                 (axis[:, :, 2]-np.cos(agent.max_tilt)).ravel(),
                 ((agent.max_axis_rate-rate)/max(agent.max_axis_rate, .1)).ravel(),
                 ((p[:, :, 2]-agent.elevation+.05)/10.).ravel(),
                 ((agent.control['throttle_rate_limit']-np.abs(slew))/
                  max(agent.control['throttle_rate_limit'], .1)).ravel()]


def repair_margins(agent, curves, durations, now, samples, safety):
    margins = constraint_margins(agent, curves, durations, now, samples)
    if safety:
        # Tighten thrust, tilt and axis-rate margins smoothly from the fixed
        # initial state. Ground clearance and the minimum-force bound retain
        # their original meaning. This never relaxes physical constraints.
        arrivals = np.cumsum(durations)
        elapsed = np.r_[0., arrivals[:-1]][:, None]+np.asarray(durations)[:, None]*np.linspace(0, 1, samples)
        guard = safety*(elapsed/arrivals[-1]).ravel()
        size = len(guard)
        for block in (0, 2, 3):
            margins[block*size:(block+1)*size] -= guard
    return margins


def repair_derivatives(agent, original, durations, now, iterations=35, safety=0.):
    """Return a dense-checked curve or None; never mutate the agent/inputs."""
    durations = np.asarray(durations, dtype=float)
    if (not np.all(np.isfinite(original)) or np.any(durations <= .15)
            or now+sum(durations) > agent.launch_time+agent.burn_time-1e-4):
        return None
    designs, _, _ = free_chain_matrices(tuple(durations), 0.)
    # Scale decision variables: 1 unit means 5 m/s or 3 m/s^2.
    scales = np.tile([5., 3.], len(durations))
    designs = designs*scales[None, None, :]
    shape = (2*len(durations), 3)
    def curves(x):
        return original+np.einsum('nki,ij->nkj', designs, x.reshape(shape))
    starts = (now+np.r_[0., np.cumsum(durations)[:-1]])[:, None]
    x = np.zeros(np.prod(shape))
    for samples, maxiter in ((25, iterations), (65, max(8, iterations//2))):
        candidate = curves(x)
        if (np.all(agent._valid(candidate, durations, starts, samples=65)[0]) and
                (not safety or np.min(repair_margins(agent, candidate, durations, now, 65, safety)) >= -1e-7)):
            return candidate
        result = minimize(lambda z: .5*float(z@z), x, jac=lambda z: z,
                          method='SLSQP', bounds=[(-6., 6.)]*len(x),
                          constraints={'type': 'ineq', 'fun': lambda z:
                              repair_margins(agent, curves(z), durations, now, samples, safety)},
                          options={'maxiter': maxiter, 'ftol': 1e-7})
        if not np.all(np.isfinite(result.x)):
            return None
        x = result.x
    candidate = curves(x)
    if (np.all(agent._valid(candidate, durations, starts, samples=65)[0]) and
            (not safety or np.min(repair_margins(agent, candidate, durations, now, 65, safety)) >= -1e-7)):
        return candidate
    return None


class ConstrainedChainAgent(ChainLaunchAgent):
    def __init__(self, given_parameters, repair_candidates=6, repair_radius=18.,
                 repair_iterations=35, repair_interval=0., **kwargs):
        super().__init__(given_parameters, **kwargs)
        self.repair_candidates = int(repair_candidates)
        self.repair_radius = float(repair_radius)
        self.repair_iterations = int(repair_iterations)
        self.repair_interval = float(repair_interval)
        if (self.repair_candidates < 0 or self.repair_iterations < 1 or
                not np.isfinite(self.repair_radius) or self.repair_radius <= 0 or
                not np.isfinite(self.repair_interval) or self.repair_interval < 0):
            raise ValueError('Invalid constrained repair settings')
        self.next_repair = 0.
        self.diagnostics.update(constraint_repairs=0, constraint_repair_accepted=0,
                                constraint_inserted_hits=0)

    def _make_plan(self, observation, position, velocity, now):
        events = len(self.route_events)
        super()._make_plan(observation, position, velocity, now)
        new_plan = len(self.route_events) != events
        if (not self.repair_candidates or not self.route or
                not new_plan and (self.repair_interval == 0 or now < self.next_repair)):
            return
        self.next_repair = now+max(self.repair_interval, .4)
        states = np.asarray(observation['balloon_states'], dtype=float)
        released = np.flatnonzero(np.asarray(observation['balloon_status']).reshape(-1) == 1)
        route = list(self.route)
        arrivals = np.asarray(self.deadlines)-now
        durations = np.diff(np.r_[0., arrivals])
        if np.min(durations) < .3 or not set(route).issubset(set(released)):
            return
        targets = states[route, :3]+states[route, 3:6]*arrivals[:, None]
        curves = free_chain(position, velocity, self.acceleration, targets, durations,
                            states[route[-1], 3:6], self.terminal_weight)
        distances, times = closest_approaches(curves, durations, states)
        candidates = [int(i) for i in released if i not in route and
                      self.failed_until.get(int(i), 0.) <= now and distances[i] <= self.repair_radius
                      and times[i] >= .5 and np.min(np.abs(arrivals-times[i])) >= .35]
        candidates.sort(key=lambda i: distances[i])
        for index in candidates[:self.repair_candidates]:
            slot = int(np.searchsorted(arrivals, times[index]))
            trial_route = route[:slot]+[index]+route[slot:]
            trial_arrivals = np.insert(arrivals, slot, times[index])
            trial_times = np.diff(np.r_[0., trial_arrivals])
            centers = states[trial_route, :3]+states[trial_route, 3:6]*trial_arrivals[:, None]
            initial = free_chain(position, velocity, self.acceleration, centers, trial_times,
                                 states[trial_route[-1], 3:6], self.terminal_weight)
            self.diagnostics['constraint_repairs'] += 1
            repaired = repair_derivatives(self, initial, trial_times, now, self.repair_iterations)
            if repaired is None:
                continue
            valid, _, vs, acs = self._valid(repaired, trial_times,
                (now+np.r_[0., trial_arrivals[:-1]])[:, None], samples=65)
            if not np.all(valid):
                continue
            self.route, self.deadlines = trial_route, list(now+trial_arrivals)
            self.ends_v, self.ends_a = list(vs), list(acs)
            self._select(trial_route[0], repaired[0], trial_times[0], now)
            event = [now, list(trial_route), float(trial_arrivals[-1])]
            if new_plan:
                self.route_events[-1] = event
            else:
                self.route_events.append(event)
            self.diagnostics['constraint_repair_accepted'] += 1
            self.diagnostics['constraint_inserted_hits'] += 1
            break
