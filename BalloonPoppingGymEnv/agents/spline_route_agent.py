"""Observation-only two-intercept minimum-jerk spline guidance.

Jointly solve intermediate velocity/acceleration, then reject physically
infeasible segments. Durations/targets use a bounded discrete local search.
"""

from functools import lru_cache

import numpy as np

from BalloonPoppingGymEnv.agents.physics_guidance_agent import PhysicsGuidanceAgent, quintic_intercept


def boundary_curve(p, v, a, end_p, end_v, end_a, duration):
    c = quintic_intercept(p, v, a, end_p, end_v, duration)
    correction = duration**2*np.asarray(end_a)
    c[3] += correction/2
    c[4] -= correction
    c[5] += correction/2
    return c


@lru_cache(maxsize=512)
def spline_matrices(first_time, second_time):
    z, one = np.zeros(1), np.ones(1)
    d1 = np.column_stack((boundary_curve(z,z,z,z,one,z,first_time)[:,0],
                          boundary_curve(z,z,z,z,z,one,first_time)[:,0]))
    d2 = np.column_stack((boundary_curve(z,one,z,z,z,z,second_time)[:,0],
                          boundary_curve(z,z,one,z,z,z,second_time)[:,0]))
    factor = np.array([6.,24.,60.])
    gram = np.outer(factor, factor)/(np.arange(3)[:,None]+np.arange(3)[None,:]+1)
    q1, q2 = gram/first_time**5, gram/second_time**5
    left, right = d1[3:].T@q1, d2[3:].T@q2
    hessian = left@d1[3:] + right@d2[3:]
    return d1, d2, left, right, np.linalg.inv(hessian)


def two_intercept_spline(p, v, a, first_p, second_p, final_v, first_time, second_time):
    """Exact minimum squared jerk over two quintics with a C2 interior knot."""
    if first_time <= 0 or second_time <= 0:
        raise ValueError("Segment durations must be positive")
    zero = np.zeros(3)
    b1 = boundary_curve(p,v,a,first_p,zero,zero,first_time)
    b2 = boundary_curve(first_p,zero,zero,second_p,final_v,zero,second_time)
    d1,d2,left,right,inverse = spline_matrices(float(first_time),float(second_time))
    interior = -inverse@(left@b1[3:]+right@b2[3:])
    return b1+d1@interior, b2+d2@interior


class SplineRouteAgent(PhysicsGuidanceAgent):
    def __init__(self, given_parameters, first_candidates=8, next_candidates=3,
                 second_horizon=8., duration_step=1., **kwargs):
        super().__init__(given_parameters, **kwargs)
        self.first_candidates = int(first_candidates)
        self.next_candidates = int(next_candidates)
        self.second_horizon = float(second_horizon)
        self.duration_step = float(duration_step)
        if min(self.first_candidates,self.next_candidates,self.second_horizon,self.duration_step) <= 0:
            raise ValueError("Search limits must be positive")
        self.diagnostics.update(pair_candidates=0, feasible_pairs=0, pair_plans=0,
                                fallback_plans=0)
        self.route_events = []

    def _make_plan(self, observation, position, velocity, now):
        states = np.asarray(observation["balloon_states"],dtype=float)
        released = np.flatnonzero(np.asarray(observation["balloon_status"]).reshape(-1)==1)
        moving = released[np.linalg.norm(states[released,3:6],axis=1)>.5]
        shared = np.median(states[moving,3:6],axis=0) if moving.size else np.zeros(3)
        drift = states[:,3:6].copy()
        drift[np.linalg.norm(drift,axis=1)<=.5] = shared
        candidates = [int(i) for i in released if self.failed_until.get(int(i),0)<=now]
        locked = self.target_index in candidates
        if locked:
            candidates = [self.target_index]
            eta = max(.4,self.plan_start+self.plan_duration-now)
            first_times = np.array([eta,eta+.8,eta+1.6])
        else:
            candidates.sort(key=lambda i: np.linalg.norm(states[i,:3]-position))
            candidates = candidates[:self.first_candidates]
            first_times = np.arange(2.,self.horizon+.01,self.duration_step)
        remaining = self.burn_time-(now-self.launch_time)-1e-5
        best = None
        for first_time in first_times:
            if first_time >= remaining:
                continue
            if best is not None and first_time+1.5 >= best[0]:
                continue
            predicted = states[:,:3]+drift*first_time
            for first in candidates:
                others = [int(j) for j in released if j != first]
                others.sort(key=lambda j: np.linalg.norm(predicted[j]-predicted[first]))
                for second in others[:self.next_candidates]:
                    for second_time in np.arange(1.5,self.second_horizon+.01,self.duration_step):
                        total = first_time+second_time
                        if total > remaining or (best is not None and total >= best[0]):
                            break
                        self.diagnostics["pair_candidates"] += 1
                        c1,c2 = two_intercept_spline(position,velocity,self.acceleration,
                            predicted[first],states[second,:3]+drift[second]*total,
                            drift[second],first_time,second_time)
                        if not self._feasible(c1,first_time,now):
                            continue
                        if not self._feasible(c2,second_time,now+first_time):
                            continue
                        self.diagnostics["feasible_pairs"] += 1
                        best = total,first,second,first_time,c1
                        break
        if best is None:
            super()._make_plan(observation,position,velocity,now)
            self.diagnostics["fallback_plans"] += 1
            return
        total,first,second,duration,curve = best
        if first != self.target_index:
            self.target_events.append((now,first))
        self.route_events.append([float(now),int(first),int(second),float(duration),float(total)])
        self.target_index = first
        self.plan,self.plan_start,self.plan_duration = curve,now,float(duration)
        self.diagnostics["plans"] += 1
        self.diagnostics["pair_plans"] += 1
