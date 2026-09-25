"""Optimize online hit points inside shrunken balloon spheres, not at centers.

Fixed timings make integrated squared jerk a convex quadratic in waypoint
offsets. Projected gradient handles the sphere constraints; dense physical
checks and backtracking protect the original feasible reference.
"""

from functools import lru_cache
import numpy as np
from BalloonPoppingGymEnv.agents.chain_launch_agent import ChainLaunchAgent
from BalloonPoppingGymEnv.agents.chain_beam_agent import free_chain


def jerk_grams(durations):
    factors = np.array([6.,24.,60.])
    gram = np.outer(factors,factors)/(np.arange(3)[:,None]+np.arange(3)[None,:]+1)
    return gram[None,:,:]/np.asarray(durations)[:,None,None]**5


def jerk_energy(curves,durations):
    return float(np.einsum('nki,nkl,nli->',curves[:,3:],jerk_grams(durations),curves[:,3:]))


@lru_cache(maxsize=256)
def offset_model(durations,terminal_weight):
    count = len(durations)
    zero = np.zeros(3)
    design = []
    for index in range(count):
        target = np.zeros((count,3))
        target[index,0] = 1.
        design.append(free_chain(zero,zero,zero,target,durations,zero,terminal_weight)[:,:,0])
    design = np.stack(design,axis=-1)
    grams = jerk_grams(durations)
    quadratic = np.einsum('nki,nkl,nlj->ij',design[:,3:],grams,design[:,3:])
    quadratic = .5*(quadratic+quadratic.T)
    return design,grams,quadratic


def optimize_offsets(curves,durations,radius,terminal_weight=0.,iterations=100):
    if radius<0 or not np.isfinite(radius):
        raise ValueError('Offset radius must be finite and nonnegative')
    design,grams,quadratic = offset_model(tuple(float(t) for t in durations),float(terminal_weight))
    linear = np.einsum('nki,nkl,nlj->ij',design[:,3:],grams,curves[:,3:])
    largest = max(float(np.linalg.eigvalsh(quadratic)[-1]),1e-12)
    offsets = np.zeros((len(durations),3))
    for _ in range(iterations):
        update = offsets-(quadratic@offsets+linear)/largest
        update *= np.minimum(1.,radius/np.maximum(np.linalg.norm(update,axis=1),1e-12))[:,None]
        if np.max(np.abs(update-offsets))<1e-7:
            offsets = update
            break
        offsets = update
    return offsets,curves+np.einsum('nki,ij->nkj',design,offsets)


class CaptureChainAgent(ChainLaunchAgent):
    def __init__(self,given_parameters,offset_fraction=.5,offset_iterations=100,**kwargs):
        super().__init__(given_parameters,**kwargs)
        self.offset_fraction = float(offset_fraction)
        self.offset_iterations = int(offset_iterations)
        if not np.isfinite(self.offset_fraction) or not 0<=self.offset_fraction<1:
            raise ValueError('offset_fraction must be in [0,1)')
        if self.offset_iterations<1:
            raise ValueError('Positive iteration count required')
        self.hit_offsets = {}
        self.diagnostics.update(offset_refinements=0,offset_accepted=0,offset_jerk_ratio_sum=0.)

    def _intercept_target(self,states,index,duration):
        return super()._intercept_target(states,index,duration)+self.hit_offsets.get(int(index),np.zeros(3))

    def _make_plan(self,observation,position,velocity,now):
        events = len(self.route_events)
        super()._make_plan(observation,position,velocity,now)
        if len(self.route_events)==events:
            return
        self.hit_offsets = {}
        if not self.route or self.offset_fraction==0:
            return
        states = np.asarray(observation['balloon_states'],dtype=float)
        route = np.asarray(self.route,dtype=int)
        durations = np.diff(np.r_[now,self.deadlines])
        if np.min(durations)<=.12:
            return
        centers = states[route,:3]+states[route,3:6]*np.cumsum(durations)[:,None]
        original = free_chain(position,velocity,self.acceleration,centers,durations,
                              states[route[-1],3:6],self.terminal_weight)
        offsets,optimized = optimize_offsets(original,durations,self.radius*self.offset_fraction,
                                             self.terminal_weight,self.offset_iterations)
        original_energy = jerk_energy(original,durations)
        self.diagnostics['offset_refinements'] += 1
        starts = (now+np.r_[0.,np.cumsum(durations)[:-1]])[:,None]
        for blend in (1.,.5,.25,.125):
            curves = original+blend*(optimized-original)
            energy = jerk_energy(curves,durations)
            if energy>=original_energy-1e-9:
                continue
            valid,_,velocities,accelerations = self._valid(curves,durations,starts,samples=65)
            if not np.all(valid):
                continue
            self.hit_offsets = {int(index):blend*offset for index,offset in zip(route,offsets)}
            self.ends_v,self.ends_a = list(velocities),list(accelerations)
            self.plan,self.plan_start,self.plan_duration = curves[0],now,float(durations[0])
            self.diagnostics['offset_accepted'] += 1
            self.diagnostics['offset_jerk_ratio_sum'] += energy/max(original_energy,1e-12)
            break
