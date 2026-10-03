"""Online route search that re-solves all arrival derivatives jointly.

Unlike appending a leg to a fixed arrival state, each new waypoint can reshape
every preceding segment before execution. Only observed released balloons are
used. The numerical model is built from given parameters and sensor history.
"""

from functools import lru_cache
import numpy as np

from BalloonPoppingGymEnv.agents.momentum_beam_agent import MomentumBeamAgent
from BalloonPoppingGymEnv.agents.submission_sqp_v2 import (
    boundary_curve, batch_samples, G,
)


@lru_cache(maxsize=2048)
def free_chain_matrices(durations, terminal_weight):
    count = len(durations)
    size = 2*count
    zero, one = np.zeros(1), np.ones(1)
    factors = np.array([6.,24.,60.])
    gram = np.outer(factors,factors)/(np.arange(3)[:,None]+np.arange(3)[None,:]+1)
    designs, projections = [], []
    hessian = np.zeros((size,size))
    for i,t in enumerate(durations):
        d = np.zeros((6,size))
        if i:
            d[:,2*(i-1)] = boundary_curve(zero,one,zero,zero,zero,zero,t)[:,0]
            d[:,2*(i-1)+1] = boundary_curve(zero,zero,one,zero,zero,zero,t)[:,0]
        d[:,2*i] = boundary_curve(zero,zero,zero,zero,one,zero,t)[:,0]
        d[:,2*i+1] = boundary_curve(zero,zero,zero,zero,zero,one,t)[:,0]
        projection = d[3:].T@(gram/t**5)
        hessian += projection@d[3:]
        designs.append(d)
        projections.append(projection)
    hessian[-2,-2] += terminal_weight
    hessian[-1,-1] += terminal_weight*4
    return np.asarray(designs), np.asarray(projections), np.linalg.inv(hessian)


def free_chain(p,v,a,targets,durations,drift,terminal_weight=0.):
    durations = tuple(float(t) for t in durations)
    zero = np.zeros(3)
    origins = [p]+list(targets[:-1])
    bases = np.array([boundary_curve(origins[i],v if i==0 else zero,
        a if i==0 else zero,targets[i],zero,zero,t) for i,t in enumerate(durations)])
    designs, projections, inverse = free_chain_matrices(durations,float(terminal_weight))
    rhs = np.einsum('nik,nkj->ij',projections,bases[:,3:])
    rhs[-2] -= terminal_weight*drift
    derivatives = -inverse@rhs
    return bases+np.einsum('nki,ij->nkj',designs,derivatives)


class ChainBeamAgent(MomentumBeamAgent):
    def __init__(self,given_parameters,terminal_weight=.1,short_leg_times=(),**kwargs):
        super().__init__(given_parameters,**kwargs)
        self.terminal_weight = float(terminal_weight)
        self.short_leg_times = np.asarray(sorted(set(float(t) for t in short_leg_times)))
        if np.any(~np.isfinite(self.short_leg_times)) or np.any((self.short_leg_times<=0)|(self.short_leg_times>=1)):
            raise ValueError('Optional short-leg times must be finite and in (0,1) seconds')

    def _candidate_ids(self, states, released, route, elapsed, position, velocity, now):
        ids = [int(i) for i in released if i not in route and self.failed_until.get(int(i),0)<=now]
        ids.sort(key=lambda i:np.linalg.norm(states[i,:3]+states[i,3:6]*(elapsed+1.5)-position-velocity*1.5))
        return ids[:self.branch_targets]

    def _node_priority(self, node, states, released, now):
        return sum(node[1])

    def _intercept_target(self, states, index, duration):
        return states[index,:3]+states[index,3:6]*duration

    def _chain_curves(self, position, velocity, targets, durations, drift):
        return free_chain(position,velocity,self.acceleration,targets,durations,
                          drift,self.terminal_weight)

    def _rescue_committed_leg(self, observation, position, velocity, now, duration):
        """Optional recovery after the active leg cannot be refreshed.

        The default deliberately preserves the established chain policy.
        Subclasses may return True only after installing a validated plan.
        """
        return False

    def _accept_committed_curve(self, curve, duration, now):
        """Check an active-leg refresh; unchanged unless a subclass opts in."""
        return self._feasible(curve, duration, now)

    def _make_plan(self,observation,position,velocity,now):
        states = np.asarray(observation['balloon_states'],dtype=float)
        released = np.flatnonzero(np.asarray(observation['balloon_status']).reshape(-1)==1)
        drift = states[:,3:6]
        while self.route and self.route[0] not in released:
            self.route.pop(0)
            self.deadlines.pop(0)
            self.ends_v.pop(0)
            self.ends_a.pop(0)
        if self.route and (self.commitment or self.route[0]==self.target_index):
            duration = self.deadlines[0]-now
            if duration>.12:
                target = self._intercept_target(states,self.route[0],duration)
                curve = boundary_curve(position,velocity,self.acceleration,target,
                                       self.ends_v[0],self.ends_a[0],duration)
                if self._accept_committed_curve(curve,duration,now):
                    self._select(self.route[0],curve,duration,now)
                    return
                if self._rescue_committed_leg(observation,position,velocity,now,duration):
                    return
                if self.plan is not None and now<self.plan_start+self.plan_duration:
                    return
            else:
                self.failed_until[self.route[0]] = now+1.
        remaining = self.launch_time+self.burn_time-now-1e-4
        initial_force = self.acceleration-G-self.disturbance
        limit = .98*self.available_acceleration(now)
        if np.linalg.norm(initial_force)>limit and limit>0:
            self.acceleration = G+self.disturbance+initial_force*limit/np.linalg.norm(initial_force)
        # Each node holds target IDs and timings, all derivatives are re-solved.
        beam = [((),(),None,position,velocity)]
        best = None
        self.diagnostics['beam_searches'] += 1
        for depth in range(self.search_depth):
            children = []
            for route,times,_,end_p,end_v in beam:
                elapsed = sum(times)
                ids = self._candidate_ids(states,released,route,elapsed,end_p,end_v,now)
                ts = np.arange(1. if depth else 2.,min(self.leg_horizon,remaining-elapsed)+.001,self.time_grid)
                if depth and len(self.short_leg_times):
                    short = self.short_leg_times[self.short_leg_times<=min(self.leg_horizon,remaining-elapsed)]
                    ts = np.r_[short,ts]
                for index in ids:
                    for t in ts:
                        ids2, durations = route+(index,), times+(float(t),)
                        arrivals = np.cumsum(durations)
                        targets = states[list(ids2),:3]+drift[list(ids2)]*arrivals[:,None]
                        curves = self._chain_curves(position,velocity,targets,durations,drift[index])
                        starts = now+np.r_[0.,arrivals[:-1]]
                        valid, ps, vs, acs = self._valid(curves,np.asarray(durations),starts[:,None],samples=17)
                        self.diagnostics['beam_candidates'] += 1
                        if not np.all(valid):
                            continue
                        self.diagnostics['beam_feasible'] += 1
                        children.append((ids2,durations,curves,ps[-1],vs[-1]))
            if not children:
                break
            children.sort(key=lambda n:self._node_priority(n,states,released,now))
            beam, counts = [], {}
            for node in children:
                key = (node[0][0],node[0][-1])
                if counts.get(key,0)>=2:
                    continue
                counts[key] = counts.get(key,0)+1
                beam.append(node)
                if len(beam)>=self.beam_width:
                    break
            best = beam[0]
            self.diagnostics['maximum_depth'] = max(self.diagnostics['maximum_depth'],depth+1)
        if best is None:
            return super()._make_plan(observation,position,velocity,now)
        route,times,curves,_,_ = best
        valid,_,vs,acs = self._valid(curves,np.asarray(times),
                                   (now+np.r_[0.,np.cumsum(times)[:-1]])[:,None],samples=65)
        if not np.all(valid):
            return super()._make_plan(observation,position,velocity,now)
        self.route,self.deadlines = list(route),list(now+np.cumsum(times))
        self.ends_v,self.ends_a = list(vs),list(acs)
        self.route_events.append([now,list(route),float(sum(times))])
        self._select(route[0],curves[0],times[0],now)
