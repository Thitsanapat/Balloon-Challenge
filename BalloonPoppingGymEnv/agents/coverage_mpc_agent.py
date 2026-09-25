"""Observation-only trajectory optimization of distinct balloon coverage.

Optimize a continuous acceleration spline. Balloon distances are measured at
their nearest simultaneous time along the path, so crossing a balloon earns
one objective term regardless of dwell time. Gaussian widths are annealed.
"""

import numpy as np
from scipy.optimize import minimize

from BalloonPoppingGymEnv.agents.submission_sqp_v2 import (
    PhysicsGuidanceAgent, G, boundary_curve, sample_curve,
)
from BalloonPoppingGymEnv.agents.momentum_beam_agent import MomentumBeamAgent


def integration_basis(knots, times):
    """Exact integrals of piecewise-linear acceleration hat functions."""
    n = len(knots)
    position = np.zeros((len(times), n))
    velocity = np.zeros_like(position)
    for i, (left, right) in enumerate(zip(knots[:-1], knots[1:])):
        h = right-left
        u = np.clip(times-left, 0, h)
        velocity[:, i] += u-u*u/(2*h)
        velocity[:, i+1] += u*u/(2*h)
        # Integrate (time-s)*phi(s) over each elapsed portion of the interval.
        position[:, i] += (times-left)*(u-u*u/(2*h))-u*u/2+u**3/(3*h)
        position[:, i+1] += (times-left)*u*u/(2*h)-u**3/(3*h)
    return position, velocity


class CoverageMPCAgent(PhysicsGuidanceAgent):
    def __init__(self, given_parameters, knot_interval=2., plan_horizon=30.,
                 solver_iterations=70, widths=(24., 12., 6., 3., 1.5),
                 thrust_margin=.97, jerk_limit=6., discount=.035,
                 warm_widths=(6., 3., 1.5), beam_seed=False, **kwargs):
        super().__init__(given_parameters, **kwargs)
        self.knot_interval = float(knot_interval)
        self.plan_horizon = float(plan_horizon)
        self.solver_iterations = int(solver_iterations)
        self.widths = tuple(float(s) for s in widths)
        self.thrust_margin = float(thrust_margin)
        self.jerk_limit = float(jerk_limit)
        self.discount = float(discount)
        self.warm_widths = tuple(float(s) for s in warm_widths)
        self.beam_seed = bool(beam_seed)
        self.previous_controls = None
        self.previous_times = None
        self.coverage_events = []
        self.diagnostics.update(coverage_solves=0, coverage_accepted=0)

    def _make_plan(self, observation, position, velocity, now):
        remaining = self.launch_time+self.burn_time-now-.02
        if remaining < .3:
            self.plan = None
            return
        states = np.asarray(observation['balloon_states'], dtype=float)
        released = np.flatnonzero(np.asarray(observation['balloon_status']).reshape(-1)==1)
        if not released.size:
            return
        horizon = min(self.plan_horizon, remaining)
        knots = np.linspace(0., horizon, max(3, int(np.ceil(horizon/self.knot_interval))+1))
        times = np.linspace(0., horizon, max(3, int(np.ceil(horizon/.2))+1))
        basis, _ = integration_basis(knots, times)
        n = len(knots)
        limits = np.array([self.available_acceleration(now+t)*self.thrust_margin for t in knots])
        ballistic = position+velocity*times[:, None]+.5*(G+self.disturbance)*times[:, None]**2
        targets = states[released, None, :3]+states[released, None, 3:6]*times[None, :, None]
        if self.previous_controls is None:
            force = np.zeros((n, 3))
            force[:, 2] = limits
        else:
            force = np.column_stack([np.interp(now+knots, self.previous_times,
                                               self.previous_controls[:, j]) for j in range(3)])
            magnitude = np.linalg.norm(force, axis=1)
            force *= np.minimum(1., limits/np.maximum(magnitude, 1e-9))[:, None]
        initial = self.acceleration-G-self.disturbance
        initial *= min(1., limits[0]/max(np.linalg.norm(initial), 1e-9))
        initial[2] = max(.1, initial[2])
        force[0] = initial
        if self.beam_seed:
            planner = MomentumBeamAgent(self.given_parameters, launch_time=self.launch_time,
                beam_width=36, search_depth=10, branch_targets=20,
                capture_fraction=0., arrival_model='relaxed')
            planner.acceleration, planner.disturbance = self.acceleration.copy(), self.disturbance.copy()
            planner._make_plan(observation, position, velocity, now)
            if planner.route:
                p, v, a, start = position.copy(), velocity.copy(), self.acceleration.copy(), now
                seeded = force.copy()
                for index, end, ev, ea in zip(planner.route, planner.deadlines, planner.ends_v, planner.ends_a):
                    duration = end-start
                    target = states[index,:3]+states[index,3:6]*(end-now)
                    curve = boundary_curve(p,v,a,target,ev,ea,duration)
                    mask = (now+knots>=start) & (now+knots<=end)
                    if np.any(mask):
                        seeded[mask] = sample_curve(curve,duration,now+knots[mask]-start)[2]-G-self.disturbance
                    p,v,a,start = target,ev,ea,end
                norm = np.linalg.norm(seeded,axis=1)
                seeded *= np.minimum(1.,limits/np.maximum(norm,1e-9))[:,None]
                seeded[:,2] = np.maximum(.1,seeded[:,2])
                seeded[0] = initial
                # Compare both initial paths using a broad capture basin.
                predicted_old = ballistic+basis@force
                predicted_new = ballistic+basis@seeded
                def seed_score(path):
                    d2 = np.sum((path[None,:,:]-targets)**2,axis=2)
                    return np.exp(-np.min(d2,axis=1)/32.).sum()
                if seed_score(predicted_new)>seed_score(predicted_old):
                    force = seeded
        dt = np.diff(knots)
        ground_jac = np.zeros((len(times), n, 3))
        ground_jac[:, :, 2] = basis
        # Norm constraints on knots imply thrust feasibility between knots up to
        # the small curvature of the known mass curve, absorbed by the margin.
        def constraint(x):
            f = x.reshape(n, 3)
            return (limits**2-np.sum(f*f, axis=1))/100.
        def constraint_jac(x):
            out = np.zeros((n, n, 3))
            out[np.arange(n), np.arange(n)] = -2*x.reshape(n, 3)/100.
            return out.reshape(n, -1)
        difference = np.zeros((n-1, n))
        difference[np.arange(n-1), np.arange(n-1)] = -1
        difference[np.arange(n-1), np.arange(1,n)] = 1
        jerk_matrix = np.kron(difference, np.eye(3))
        jerk_bounds = np.repeat(self.jerk_limit*dt, 3)
        constraints = [
            {'type':'ineq', 'fun':constraint, 'jac':constraint_jac},
            {'type':'ineq', 'fun':lambda x:ballistic[:, 2]+basis@x.reshape(n,3)[:,2]-self.elevation+.04,
             'jac':lambda x:ground_jac.reshape(len(times), -1)},
            {'type':'ineq', 'fun':lambda x:np.r_[jerk_bounds-jerk_matrix@x, jerk_bounds+jerk_matrix@x],
             'jac':lambda x:np.r_[-jerk_matrix, jerk_matrix]},
        ]
        bounds = [(-float(l), float(l)) if j<2 else (.1, float(l)) for l in limits for j in range(3)]
        bounds[:3] = [(float(x),float(x)) for x in initial]
        def closest(x, width=0.):
            path = ballistic+basis@x.reshape(n,3)
            relative = path[None, :, :]-targets
            a, delta = relative[:, :-1], np.diff(relative, axis=1)
            fraction = np.clip((-np.sum(a*delta, axis=2)-width**2*self.discount*np.diff(times))
                               /np.maximum(np.sum(delta*delta,axis=2),1e-12),0,1)
            errors = a+fraction[:, :, None]*delta
            distances = np.sum(errors*errors,axis=2)
            arrivals = times[:-1]+fraction*np.diff(times)
            idx = np.argmin(distances+2*width**2*self.discount*arrivals,axis=1)
            rows = np.arange(len(idx))
            u = fraction[rows,idx]
            weights = basis[idx]*(1-u[:,None])+basis[idx+1]*u[:,None]
            return distances[rows,idx], errors[rows,idx], weights, arrivals[rows,idx]
        x = force.ravel()
        for width in (self.widths if self.previous_controls is None else self.warm_widths):
            def objective_and_jac(z):
                distance2, error, weights, arrivals = closest(z, width)
                reward = np.exp(-distance2/(2*width**2)-self.discount*arrivals)
                # Each observed balloon contributes at most one hit.
                cost = -float(reward.sum())
                gradient = np.einsum('b,bi,bj->ij', reward/width**2, weights, error)
                return cost, gradient.ravel()
            solution = minimize(objective_and_jac, x, jac=True, method='SLSQP',
                bounds=bounds, constraints=constraints,
                options={'maxiter':self.solver_iterations, 'ftol':1e-5})
            self.diagnostics['coverage_solves'] += 1
            if np.all(np.isfinite(solution.x)) and min(np.min(c['fun'](solution.x)) for c in constraints)>-1e-5:
                x = solution.x
                self.diagnostics['coverage_accepted'] += 1
        force = x.reshape(n,3)
        duration = float(knots[1])
        acceleration0 = force[0]+G+self.disturbance
        acceleration1 = force[1]+G+self.disturbance
        jerk = (acceleration1-acceleration0)/duration
        end_p = position+velocity*duration+acceleration0*duration**2/2+jerk*duration**3/6
        end_v = velocity+acceleration0*duration+jerk*duration**2/2
        self.plan = boundary_curve(position, velocity, acceleration0, end_p,
                                   end_v, acceleration1, duration)
        self.plan_start, self.plan_duration, self.target_index = now, duration, None
        self.previous_controls, self.previous_times = force.copy(), now+knots
        distance2, _, _, _ = closest(x)
        self.coverage_events.append([now, int(np.sum(distance2<self.radius**2))])
        self.diagnostics['plans'] += 1
