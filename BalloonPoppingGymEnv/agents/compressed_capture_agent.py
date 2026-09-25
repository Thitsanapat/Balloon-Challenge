"""Spend capture-region slack on shorter subsequent intercept legs, if feasible."""

import numpy as np
from BalloonPoppingGymEnv.agents.capture_chain_agent import CaptureChainAgent, optimize_offsets
from BalloonPoppingGymEnv.agents.chain_beam_agent import free_chain


class CompressedCaptureAgent(CaptureChainAgent):
    def __init__(self,given_parameters,time_compression=.06,**kwargs):
        super().__init__(given_parameters,**kwargs)
        self.time_compression = float(time_compression)
        if not np.isfinite(self.time_compression) or not 0<=self.time_compression<=.2:
            raise ValueError('time_compression must lie in [0,.2]')
        self.diagnostics.update(compression_attempts=0,compression_accepted=0,
                                compressed_reference_seconds=0.)

    def _make_plan(self,observation,position,velocity,now):
        events = len(self.route_events)
        super()._make_plan(observation,position,velocity,now)
        if len(self.route_events)==events or len(self.route)<2 or self.time_compression==0:
            return
        initial = np.diff(np.r_[now,self.deadlines])
        if np.min(initial)<=.12:
            return
        states = np.asarray(observation['balloon_states'],dtype=float)
        route = np.asarray(self.route,dtype=int)
        self.diagnostics['compression_attempts'] += 1
        for fraction in (self.time_compression,self.time_compression*.5,self.time_compression*.25):
            durations = initial.copy()
            # Preserve first-hit timing, which often uses almost all vertical
            # thrust. Only shorten the later legs; re-solve all knot derivatives.
            durations[1:] *= 1-fraction
            centers = states[route,:3]+states[route,3:6]*np.cumsum(durations)[:,None]
            original = free_chain(position,velocity,self.acceleration,centers,durations,
                                  states[route[-1],3:6],self.terminal_weight)
            offsets,optimized = optimize_offsets(original,durations,self.radius*self.offset_fraction,
                                                 self.terminal_weight,self.offset_iterations)
            starts = (now+np.r_[0.,np.cumsum(durations)[:-1]])[:,None]
            for blend in (1.,.5):
                curves = original+blend*(optimized-original)
                valid,_,velocities,accelerations = self._valid(curves,durations,starts,samples=65)
                if not np.all(valid):
                    continue
                self.deadlines = list(now+np.cumsum(durations))
                self.hit_offsets = {int(index):blend*offset for index,offset in zip(route,offsets)}
                self.ends_v,self.ends_a = list(velocities),list(accelerations)
                self.plan,self.plan_start,self.plan_duration = curves[0],now,float(durations[0])
                self.diagnostics['compression_accepted'] += 1
                self.diagnostics['compressed_reference_seconds'] += float(np.sum(initial-durations))
                return
