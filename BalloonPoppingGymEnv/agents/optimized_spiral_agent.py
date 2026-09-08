"""Local constrained trajectory optimization with a soft wide-spiral reference.

Discrete observed target selection + SLSQP arrival-time/velocity optimization.
No global optimality, full-helix tracking, or continuous feasibility claim.
"""

import numpy as np
from scipy.optimize import minimize

from BalloonPoppingGymEnv.agents.physics_guidance_agent import G, quintic_intercept, sample_curve
from BalloonPoppingGymEnv.agents.wind_spiral_agent import WindSpiralAgent, spiral_state


def jerk_energy(curve, duration):
    """Exact integral of squared jerk for normalized-time quintic coefficients."""
    j = curve[3:] * np.array([6., 24., 60.])[:, None]
    gram = 1. / (np.arange(3)[:, None]+np.arange(3)[None, :]+1.)
    return float(np.sum((j@j.T)*gram)/duration**5)


class OptimizedSpiralAgent(WindSpiralAgent):
    def __init__(self, given_parameters, spiral_weight=0.4, jerk_weight=0.002,
                 optimization_interval=2., **kwargs):
        super().__init__(given_parameters, **kwargs)
        self.spiral_weight = float(spiral_weight)
        self.jerk_weight = float(jerk_weight)
        self.optimization_interval = float(optimization_interval)
        if min(self.spiral_weight, self.jerk_weight, self.optimization_interval) < 0:
            raise ValueError("Optimization weights/interval must be nonnegative")
        self.next_optimization = 0.
        self.terminal_offset = np.zeros(3)
        self.diagnostics.update(optimizer_calls=0, optimizer_improvements=0,
                                optimizer_failures=0, optimized_intercept_plans=0)

    def _constraint_margins(self, curve, duration, now):
        times = np.linspace(0., duration, 33)
        p, v, a, jerk = sample_curve(curve, duration, times)
        force = a-G-self.disturbance
        magnitude = np.linalg.norm(force, axis=1)
        limits = np.array([self.available_acceleration(now+t) for t in times])
        axes = force/np.maximum(magnitude[:, None], 1e-9)
        rate = np.linalg.norm(jerk-axes*np.sum(axes*jerk, axis=1)[:, None], axis=1)/np.maximum(magnitude, 1e-9)
        throttle = magnitude/np.maximum(limits, 1e-9)
        return np.concatenate(((limits-magnitude)/12., (magnitude-0.5)/12.,
                               axes[:, 2]-np.cos(self.max_tilt), self.max_axis_rate-rate,
                               self.control["throttle_rate_limit"]-np.abs(np.diff(throttle)/np.diff(times)),
                               (p[:, 2]-self.elevation+0.05)/10.))

    def _objective(self, curve, duration, geometry, deadline=None):
        samples = np.linspace(0., duration, 9)
        p = sample_curve(curve, duration, samples)[0]
        reference = np.array([spiral_state(*geometry, t)[0] for t in samples])
        deviation = np.mean(np.sum((p-reference)**2, axis=1))/max(geometry[4]**2, 1.)
        delay = 0. if deadline is None else 4.*max(0., duration-deadline)
        return float(duration + self.spiral_weight*deviation
                     + self.jerk_weight*jerk_energy(curve, duration) + delay)

    def _make_plan(self, observation, position, velocity, now):
        geometry = self._spiral_geometry(observation, position, now)
        if geometry is None:
            # Missing drift only: use the existing observation-only planner.
            from BalloonPoppingGymEnv.agents.physics_guidance_agent import PhysicsGuidanceAgent
            PhysicsGuidanceAgent._make_plan(self, observation, position, velocity, now)
            return
        states = np.asarray(observation["balloon_states"], dtype=float)
        released = np.flatnonzero(np.asarray(observation["balloon_status"]).reshape(-1) == 1)
        candidates = [int(i) for i in released if self.failed_until.get(int(i), 0) <= now]
        locked = self.target_index in candidates
        if locked:
            candidates = [self.target_index]
        else:
            candidates.sort(key=lambda i: np.linalg.norm(states[i, :3]-position))
            candidates = candidates[:12]
        horizon = min(self.horizon, self.burn_time-(now-self.launch_time)-1e-5)
        deadline = max(0.4, self.plan_start+self.plan_duration-now) if locked else None
        earliest = deadline if locked else 0.8
        seeds = []
        for index in candidates:
            drift = states[index, 3:6]
            if np.linalg.norm(drift) < 0.5:
                drift = self.filtered_drift
            tangent = np.cross(np.cross(geometry[2], geometry[3]), states[index, :3]-geometry[0])
            tangent /= max(np.linalg.norm(tangent), 1e-9)
            offsets = [self.terminal_offset] if locked else [np.zeros(3), 4.*tangent]
            for offset in offsets:
                for duration in np.unique(np.r_[earliest, np.arange(max(earliest, 0.8), horizon+0.001, 0.4)]):
                    if duration > horizon:
                        continue
                    curve = quintic_intercept(position, velocity, self.acceleration,
                                              states[index, :3]+drift*duration, drift+offset, duration)
                    if self._feasible(curve, duration, now):
                        cost = self._objective(curve, duration, geometry, deadline)
                        seeds.append((cost, index, np.r_[duration, offset], curve, drift))
                        break
        seeds.sort(key=lambda item: item[0])
        if not seeds:
            self.diagnostics["no_feasible_plan"] += 1
            if self.plan is not None and now < self.plan_start+self.plan_duration:
                return
            if self.target_index is not None:
                self.failed_until[self.target_index] = now+2*self.replan_interval
            self.target_index, self.plan = None, None
            return
        best = seeds[0]
        if not locked or now >= self.next_optimization:
            for seed in seeds[:1 if locked else 3]:
                cost, index, x0, curve, drift = seed
                def make_curve(x):
                    return quintic_intercept(position, velocity, self.acceleration,
                                             states[index, :3]+drift*x[0], drift+x[1:], x[0])
                def objective(x):
                    return self._objective(make_curve(x), x[0], geometry, deadline)
                self.diagnostics["optimizer_calls"] += 1
                result = minimize(objective, x0, method="SLSQP",
                                  bounds=[(0.4, horizon)]+[(-6., 6.)]*3,
                                  # Leave numerical slack: SLSQP's tolerance must
                                  # not place a nominal result just outside the
                                  # strict inherited feasibility checks.
                                  constraints={"type": "ineq", "fun": lambda x: self._constraint_margins(make_curve(x), x[0], now)-1e-5},
                                  options={"maxiter": 15, "ftol": 1e-4})
                # Solver success alone does not certify a usable trajectory.
                candidate_curve = make_curve(result.x)
                candidate_cost = objective(result.x)
                if np.all(np.isfinite(result.x)) and self._feasible(candidate_curve, result.x[0], now) and candidate_cost < best[0]-1e-6:
                    best = (candidate_cost, index, result.x.copy(), candidate_curve, drift)
                    self.diagnostics["optimizer_improvements"] += 1
                if not result.success:
                    self.diagnostics["optimizer_failures"] += 1
            self.next_optimization = now+self.optimization_interval
        _, index, x, curve, _ = best
        if index != self.target_index:
            self.target_events.append((now, index))
        self.target_index = index
        self.terminal_offset = x[1:].copy()
        self.plan, self.plan_start, self.plan_duration = curve, now, float(x[0])
        self.diagnostics["plans"] += 1
        self.diagnostics["optimized_intercept_plans"] += 1
