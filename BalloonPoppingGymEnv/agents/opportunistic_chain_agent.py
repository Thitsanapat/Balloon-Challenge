"""Insert observation-predicted near-path hits into a feasible online chain."""

import numpy as np
from BalloonPoppingGymEnv.agents.chain_launch_agent import ChainLaunchAgent
from BalloonPoppingGymEnv.agents.chain_beam_agent import free_chain
from BalloonPoppingGymEnv.agents.capture_chain_agent import optimize_offsets
from BalloonPoppingGymEnv.agents.submission_sqp_v2 import batch_samples


def closest_approaches(curves,durations,states,samples=33):
    """Piecewise-linear relative-motion proximity; a planning approximation."""
    positions,_,_,_ = batch_samples(curves,durations,samples)
    ts = np.r_[0.,np.cumsum(durations)[:-1]][:,None]+np.asarray(durations)[:,None]*np.linspace(0,1,samples)
    points,times = positions.reshape(-1,3),ts.reshape(-1)
    relative = points[:,None,:]-states[None,:,:3]-times[:,None,None]*states[None,:,3:6]
    delta = np.diff(relative,axis=0)
    alpha = np.clip(-np.sum(relative[:-1]*delta,axis=2)/np.maximum(np.sum(delta*delta,axis=2),1e-12),0.,1.)
    distance = np.linalg.norm(relative[:-1]+alpha[:,:,None]*delta,axis=2)
    closest = np.argmin(distance,axis=0)
    columns = np.arange(len(states))
    return distance[closest,columns],times[closest]+alpha[closest,columns]*np.diff(times)[closest]


class OpportunisticChainAgent(ChainLaunchAgent):
    def __init__(self,given_parameters,insertion_radius=5.,max_insertions=2,
                 insertion_candidates=8,insertion_offset_fraction=.5,insertion_interval=0.,**kwargs):
        super().__init__(given_parameters,**kwargs)
        self.insertion_radius = float(insertion_radius)
        self.max_insertions = int(max_insertions)
        self.insertion_candidates = int(insertion_candidates)
        self.insertion_offset_fraction = float(insertion_offset_fraction)
        self.insertion_interval = float(insertion_interval)
        self.next_insertion_check = 0.
        if not np.isfinite(self.insertion_interval) or self.insertion_interval<0:
            raise ValueError('Nonnegative finite insertion interval required')
        if not np.isfinite(self.insertion_radius) or self.insertion_radius<=0:
            raise ValueError('Positive finite insertion radius required')
        if min(self.max_insertions,self.insertion_candidates)<0 or not 0<=self.insertion_offset_fraction<1:
            raise ValueError('Invalid insertion limits')
        self.hit_offsets = {}
        self.diagnostics.update(insertion_attempts=0,inserted_waypoints=0)

    def _intercept_target(self,states,index,duration):
        return super()._intercept_target(states,index,duration)+self.hit_offsets.get(int(index),np.zeros(3))

    def _make_plan(self,observation,position,velocity,now):
        events = len(self.route_events)
        super()._make_plan(observation,position,velocity,now)
        new_plan = len(self.route_events)!=events
        if not new_plan and (self.insertion_interval==0 or now<self.next_insertion_check):
            return
        if new_plan:
            self.hit_offsets = {}
        self.next_insertion_check = now+self.insertion_interval
        if not self.route or not self.max_insertions:
            return
        states = np.asarray(observation['balloon_states'],dtype=float)
        released = np.flatnonzero(np.asarray(observation['balloon_status']).reshape(-1)==1)
        route = list(self.route)
        arrivals = np.asarray(self.deadlines)-now
        durations = np.diff(np.r_[0.,arrivals])
        if np.min(durations)<=.15:
            return
        targets = states[route,:3]+states[route,3:6]*arrivals[:,None]
        targets += np.asarray([self.hit_offsets.get(i,np.zeros(3)) for i in route])
        curves = free_chain(position,velocity,self.acceleration,targets,durations,states[route[-1],3:6],self.terminal_weight)
        for _ in range(self.max_insertions):
            distances,times = closest_approaches(curves,durations,states)
            candidates = [int(i) for i in released if i not in route and distances[i]<=self.insertion_radius
                          and times[i]>.2 and np.min(np.abs(arrivals-times[i]))>.2]
            candidates.sort(key=lambda i:distances[i])
            accepted = False
            for index in candidates[:self.insertion_candidates]:
                slot = int(np.searchsorted(arrivals,times[index]))
                trial_route = route[:slot]+[index]+route[slot:]
                trial_arrivals = np.insert(arrivals,slot,times[index])
                trial_durations = np.diff(np.r_[0.,trial_arrivals])
                centers = states[trial_route,:3]+states[trial_route,3:6]*trial_arrivals[:,None]
                original = free_chain(position,velocity,self.acceleration,centers,trial_durations,
                                      states[trial_route[-1],3:6],self.terminal_weight)
                offsets,optimized = optimize_offsets(original,trial_durations,self.radius*self.insertion_offset_fraction,
                                                     self.terminal_weight)
                starts = (now+np.r_[0.,trial_arrivals[:-1]])[:,None]
                self.diagnostics['insertion_attempts'] += 1
                for blend in (0.,1.,.5):
                    trial = original+blend*(optimized-original)
                    valid,_,vs,acs = self._valid(trial,trial_durations,starts,samples=65)
                    if not np.all(valid):
                        continue
                    route,arrivals,durations,curves = trial_route,trial_arrivals,trial_durations,trial
                    self.route,self.deadlines = list(route),list(now+arrivals)
                    self.ends_v,self.ends_a = list(vs),list(acs)
                    self.hit_offsets = {int(i):blend*d for i,d in zip(route,offsets)}
                    self._select(route[0],curves[0],durations[0],now)
                    event = [now,list(route),float(arrivals[-1])]
                    if new_plan:
                        self.route_events[-1] = event
                    else:
                        self.route_events.append(event)
                        new_plan = True
                    self.diagnostics['inserted_waypoints'] += 1
                    accepted = True
                    break
                if accepted:
                    break
            if not accepted:
                break
