"""Extend the best two-target plan with a jointly smoothed third intercept."""

from functools import lru_cache

import numpy as np

from BalloonPoppingGymEnv.agents.spline_route_agent import SplineRouteAgent, boundary_curve


@lru_cache(maxsize=512)
def chain_matrices(durations):
    count = len(durations)
    unknowns = 2*(count-1)
    z, one = np.zeros(1), np.ones(1)
    factors = np.array([6.,24.,60.])
    gram = np.outer(factors,factors)/(np.arange(3)[:,None]+np.arange(3)[None,:]+1)
    designs, projections = [], []
    hessian = np.zeros((unknowns,unknowns))
    for i,duration in enumerate(durations):
        design = np.zeros((6,unknowns))
        if i > 0:
            design[:,2*(i-1)] = boundary_curve(z,one,z,z,z,z,duration)[:,0]
            design[:,2*(i-1)+1] = boundary_curve(z,z,one,z,z,z,duration)[:,0]
        if i < count-1:
            design[:,2*i] = boundary_curve(z,z,z,z,one,z,duration)[:,0]
            design[:,2*i+1] = boundary_curve(z,z,z,z,z,one,duration)[:,0]
        projection = design[3:].T@(gram/duration**5)
        hessian += projection@design[3:]
        designs.append(design)
        projections.append(projection)
    return designs,projections,np.linalg.inv(hessian)


def intercept_chain(position,velocity,acceleration,waypoints,final_velocity,durations):
    """Minimum total squared jerk, with free internal velocities/accelerations."""
    durations = tuple(float(t) for t in durations)
    if len(durations)<2 or len(waypoints)!=len(durations) or not all(np.isfinite(t) and t>0 for t in durations):
        raise ValueError("Supply matching waypoints and at least two positive durations")
    zero = np.zeros(3)
    origins = [position]+list(waypoints[:-1])
    bases = [boundary_curve(origins[i],velocity if i==0 else zero,
                            acceleration if i==0 else zero,waypoints[i],
                            final_velocity if i==len(durations)-1 else zero,
                            zero,t) for i,t in enumerate(durations)]
    designs,projections,inverse = chain_matrices(durations)
    rhs = sum(projection@base[3:] for projection,base in zip(projections,bases))
    interior = -inverse@rhs
    return [base+design@interior for base,design in zip(bases,designs)]


class ThreeInterceptAgent(SplineRouteAgent):
    def __init__(self,given_parameters,third_candidates=4,third_horizon=8.,**kwargs):
        super().__init__(given_parameters,**kwargs)
        self.third_candidates = int(third_candidates)
        self.third_horizon = float(third_horizon)
        if self.third_candidates<1 or self.third_horizon<1.5:
            raise ValueError("Invalid third-segment search limits")
        self.diagnostics.update(triple_candidates=0,feasible_triples=0,triple_plans=0)
        self.triple_events = []

    def _make_plan(self,observation,position,velocity,now):
        previous_pairs = self.diagnostics["pair_plans"]
        super()._make_plan(observation,position,velocity,now)
        if self.diagnostics["pair_plans"]==previous_pairs:
            return
        _,first,second,first_time,pair_time = self.route_events[-1]
        states = np.asarray(observation["balloon_states"],dtype=float)
        released = np.flatnonzero(np.asarray(observation["balloon_status"]).reshape(-1)==1)
        moving = released[np.linalg.norm(states[released,3:6],axis=1)>.5]
        drift = states[:,3:6].copy()
        if moving.size:
            drift[np.linalg.norm(drift,axis=1)<=.5] = np.median(drift[moving],axis=0)
        remaining = self.burn_time-(now-self.launch_time)-1e-5
        projected = states[:,:3]+drift*pair_time
        candidates = [int(i) for i in released if i not in (first,second)]
        candidates.sort(key=lambda i: np.linalg.norm(projected[i]-projected[second]))
        best = None
        # Preserve the first arrival deadline, while allowing half a second of
        # adjustment to the second leg. First/second target choice stays bounded
        # by the existing pair search rather than an exponential route tree.
        for second_time in sorted(set([max(.5,pair_time-first_time-.5),pair_time-first_time,pair_time-first_time+.5])):
            for third in candidates[:self.third_candidates]:
                for third_time in np.arange(1.5,self.third_horizon+.01,self.duration_step):
                    total = first_time+second_time+third_time
                    if total>remaining or (best is not None and total>=best[0]):
                        break
                    durations = [first_time,second_time,third_time]
                    arrival = np.cumsum(durations)
                    ids = [first,second,third]
                    targets = [states[i,:3]+drift[i]*t for i,t in zip(ids,arrival)]
                    curves = intercept_chain(position,velocity,self.acceleration,targets,
                                             drift[third],durations)
                    self.diagnostics["triple_candidates"] += 1
                    starts = np.r_[0.,arrival[:-1]]
                    if all(self._feasible(c,t,now+offset) for c,t,offset in zip(curves,durations,starts)):
                        self.diagnostics["feasible_triples"] += 1
                        best = total,curves[0],third,second_time,third_time
                        break
        if best is not None:
            total,curve,third,second_time,third_time = best
            self.plan = curve
            self.diagnostics["triple_plans"] += 1
            self.triple_events.append([float(now),first,second,third,first_time,second_time,third_time])
