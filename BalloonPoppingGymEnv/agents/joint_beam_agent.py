"""Minimum-time refinement of quintic routes seeded by feasible fly-throughs."""

import numpy as np
from scipy.optimize import minimize

from BalloonPoppingGymEnv.agents.beam_intercept_agent import BeamInterceptAgent, batch_samples
from BalloonPoppingGymEnv.agents.physics_guidance_agent import G, sample_curve
from BalloonPoppingGymEnv.agents.spline_route_agent import boundary_curve


class JointBeamAgent(BeamInterceptAgent):
    def __init__(self, given_parameters, solver_iterations=30, optimize_derivatives=True, **kwargs):
        super().__init__(given_parameters, **kwargs)
        self.solver_iterations = int(solver_iterations)
        self.optimize_derivatives = bool(optimize_derivatives)
        self.committed_route = []
        self.deadlines = np.array([])
        self.endpoint_velocities = self.endpoint_accelerations = None
        self.diagnostics.update(joint_calls=0, joint_accepted=0, route_continuations=0)

    def _curves(self, position, velocity, states, drift, route, durations):
        arrivals = np.cumsum(durations)
        targets = states[route, :3]+drift[route]*arrivals[:, None]
        curves = []
        p, v, a = position, velocity, self.acceleration
        for target, end_v, end_a, duration in zip(
                targets, self.endpoint_velocities, self.endpoint_accelerations, durations):
            curves.append(boundary_curve(p, v, a, target, end_v, end_a, duration))
            p, v, a = target, end_v, end_a
        return np.asarray(curves)

    def _margins(self, curves, durations, now, samples=33):
        durations = np.asarray(durations, dtype=float)
        p, _, a, j = batch_samples(curves, durations, samples)
        starts = np.r_[0., np.cumsum(durations)[:-1]]
        times = now+starts[:, None]+durations[:, None]*np.linspace(0., 1., samples)
        age = np.maximum(times-self.launch_time, 0.)
        limit = self.thrust/np.maximum(self.dry_mass, self.initial_mass-self.mass_flow*age)
        force = a-G-self.disturbance
        mag = np.linalg.norm(force, axis=2)
        axis = force/np.maximum(mag[:, :, None], 1e-9)
        rates = np.linalg.norm(j-axis*np.sum(axis*j, axis=2)[:, :, None], axis=2)/np.maximum(mag, 1e-9)
        throttle = mag/limit
        throttle_change = np.diff(throttle, axis=1)/np.diff(times, axis=1)
        return np.r_[(limit-mag).ravel()/12., (mag-.5).ravel()/12.,
                     (axis[:, :, 2]-np.cos(self.max_tilt)).ravel(),
                     (self.max_axis_rate**2-rates**2).ravel(),
                     (p[:, :, 2]-self.elevation+.05).ravel()/10.,
                     (self.control['throttle_rate_limit']**2-throttle_change**2).ravel(),
                     self.launch_time+self.burn_time-now-sum(durations)-1e-4]

    def _solve(self, position, velocity, states, drift, route, initial, now, held):
        self.diagnostics['joint_calls'] += 1
        count = len(initial)
        free = self.optimize_derivatives and not held
        seed_v, seed_a = self.endpoint_velocities.copy(), self.endpoint_accelerations.copy()
        seed = np.r_[initial, seed_v.ravel()/10., seed_a.ravel()/5.] if free else initial
        def unpack(x):
            if free:
                self.endpoint_velocities = x[count:count+3*count].reshape(count, 3)*10.
                self.endpoint_accelerations = x[count+3*count:].reshape(count, 3)*5.
            return x[:count]
        def curves(x):
            durations = unpack(x)
            return self._curves(position, velocity, states, drift, route, durations)
        def margins(x):
            return self._margins(curves(x), x[:count], now)
        def objective(x):
            # Penalize postponing the current hit while still improving the tail.
            return sum(x[:count])+(4.*max(0., x[0]-initial[0]) if held else 0.)
        bounds = [(max(.15, t-2.), t+2.) for t in initial]
        if free:
            bounds += [(-6., 6.)]*(3*count)+[(-2., 2.)]*(3*count)
        result = minimize(objective, seed, method='SLSQP',
                          bounds=bounds,
                          constraints={'type':'ineq', 'fun':margins},
                          options={'maxiter':self.solver_iterations, 'ftol':1e-8})
        choices = [seed]
        if np.all(np.isfinite(result.x)):
            # An optimum at sampled constraint boundaries can violate them
            # between samples. Backtrack toward the feasible seed, then check
            # every candidate at the denser resolution before execution.
            choices.extend(seed+fraction*(result.x-seed)
                           for fraction in (1., .99, .97, .95, .9, .8, .6, .4))
        choices.sort(key=objective)
        for x in choices:
            if np.min(margins(x)) < -1e-6:
                continue
            cs = curves(x)
            durations = x[:count]
            # Independent denser check before accepting the optimized plan.
            # Tiny numerical tolerance, not extra physical control authority.
            # Check twice the optimizer's temporal resolution for missed extrema.
            if np.min(self._margins(cs, durations, now, samples=65)) >= -1e-6:
                self.diagnostics['joint_accepted'] += 1
                return cs, np.asarray(durations)
        self.endpoint_velocities, self.endpoint_accelerations = seed_v, seed_a
        return None

    def _make_plan(self, observation, position, velocity, now):
        states = np.asarray(observation['balloon_states'], dtype=float)
        status = np.asarray(observation['balloon_status']).reshape(-1)
        drift = states[:, 3:6].copy()
        moving = np.flatnonzero((status==1) & (np.linalg.norm(drift, axis=1)>.5))
        if moving.size:
            drift[np.linalg.norm(drift, axis=1)<=.5] = np.median(drift[moving], axis=0)
        while self.committed_route and status[self.committed_route[0]] != 1:
            self.committed_route.pop(0)
            self.deadlines = self.deadlines[1:]
            self.endpoint_velocities = self.endpoint_velocities[1:]
            self.endpoint_accelerations = self.endpoint_accelerations[1:]
        held = bool(self.committed_route and self.deadlines[0]>now+.15)
        route = self.committed_route.copy() if held else []
        if held:
            initial = np.diff(np.r_[now, self.deadlines])
        else:
            self.committed_route = []
            previous = len(self.route_events)
            super()._make_plan(observation, position, velocity, now)
            if len(self.route_events)==previous:
                return
            route = list(self.route_events[-1][1])
            initial = np.asarray(self.selected_leg_times)
            _, vs, acs, _ = batch_samples(self.selected_leg_curves, initial)
            self.endpoint_velocities = vs[:, -1].copy()
            self.endpoint_accelerations = acs[:, -1].copy()
        if len(route)<2:
            self.committed_route = []
            super()._make_plan(observation, position, velocity, now)
            return
        solved = self._solve(position, velocity, states, drift, route, initial, now, held)
        if solved is None:
            if held:
                super()._make_plan(observation, position, velocity, now)
            return
        cs, durations = solved
        self.committed_route = route
        self.deadlines = now+np.cumsum(durations)
        if route[0] != self.target_index:
            self.target_events.append((now, route[0]))
        self.target_index = route[0]
        self.plan, self.plan_start, self.plan_duration = cs[0], now, float(durations[0])
        _, self.end_velocity, self.end_acceleration, _ = sample_curve(cs[0], durations[0], durations[0])
        self.diagnostics['plans'] += 1
        self.diagnostics['route_continuations'] += int(held)
