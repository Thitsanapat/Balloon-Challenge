"""Observation-only interception with a short acceleration ramp and coast.

Acceleration changes over a bounded slew interval, then stays constant until
the intercept. Search carries the resulting velocity into subsequent targets.
This avoids spreading every attitude change over an entire polynomial leg.
"""

import numpy as np
from BalloonPoppingGymEnv.agents.submission_sqp_v2 import PhysicsGuidanceAgent, G


def ramp_solution(p,v,a,targets,durations,ramps):
    t,r = durations[:,None],ramps[:,None]
    k = t*r/2-r*r/6
    end_a = (targets-p-v*t-a*k)/(.5*t*t-k)
    end_v = v+a*r/2+end_a*(t-r/2)
    return end_a,end_v


def ramp_reference(p,v,a,end_a,duration,ramp):
    ramp = min(ramp,duration)
    delta = end_a-a
    first = np.zeros((6,3))
    first[:4] = p,v*ramp,a*ramp**2/2,delta*ramp**2/6
    mid_p = p+v*ramp+a*ramp**2/2+delta*ramp**2/6
    mid_v = v+(a+end_a)*ramp/2
    rest = max(duration-ramp,1e-6)
    second = np.zeros((6,3))
    second[:3] = mid_p,mid_v*rest,end_a*rest**2/2
    return first,second


class RampBeamAgent(PhysicsGuidanceAgent):
    def __init__(self,given_parameters,beam_width=48,search_depth=12,
                 branch_targets=20,time_grid=.5,leg_horizon=10.,
                 ramp_options=(.4,.7,1.1),reserve=.15,**kwargs):
        super().__init__(given_parameters,**kwargs)
        self.beam_width,self.search_depth = int(beam_width),int(search_depth)
        self.branch_targets = int(branch_targets)
        self.time_grid,self.leg_horizon = float(time_grid),float(leg_horizon)
        self.ramp_options = np.asarray(ramp_options,dtype=float)
        self.reserve = float(reserve)
        self.active = None
        self.route_events = []
        self.diagnostics.update(beam_searches=0,beam_candidates=0,beam_feasible=0,maximum_depth=0)

    def _valid(self,p,v,a,ad,ts,rs,start):
        samples = np.linspace(0,1,25)
        elapsed = ts[:,None]*samples
        u = np.minimum(elapsed,rs[:,None])
        delta = ad-a
        # Integrate the linear ramp, then the constant acceleration section.
        tail = np.maximum(0,elapsed-rs[:,None])
        ps = (p+v*elapsed[:,:,None]+a*u[:,:,None]**2/2
              +delta[:,None,:]*u[:,:,None]**3/(6*rs[:,None,None])
              +(a*rs[:,None]/2+ad*rs[:,None]/2)[:,None,:]*tail[:,:,None]
              +ad[:,None,:]*tail[:,:,None]**2/2)
        ac = a+delta[:,None,:]*(u/rs[:,None])[:,:,None]
        force = ac-G-self.disturbance
        mag = np.linalg.norm(force,axis=2)
        age = np.maximum(0,start-self.launch_time+elapsed)
        limits = self.thrust/np.maximum(self.dry_mass,self.initial_mass-self.mass_flow*age)
        # Check the ramp separately so short slew intervals cannot hide between samples.
        ramp_s = np.linspace(0,1,9)
        rf = a-G-self.disturbance+delta[:,None,:]*ramp_s[None,:,None]
        rm = np.linalg.norm(rf,axis=2)
        axis = rf/np.maximum(rm[:,:,None],1e-9)
        jerk = delta[:,None,:]/rs[:,None,None]
        rates = np.linalg.norm(jerk-axis*np.sum(axis*jerk,axis=2)[:,:,None],axis=2)/np.maximum(rm,1e-9)
        valid = np.all((mag<=limits-self.reserve*samples+1e-6)
                       &(mag>.5)&(force[:,:,2]>=mag*np.cos(self.max_tilt))
                       &(ps[:,:,2]>=self.elevation-.05),axis=1)
        valid &= np.all(rates<=self.max_axis_rate,axis=1)
        ramp_limit = self.thrust/np.maximum(self.dry_mass,self.initial_mass-self.mass_flow*
                        np.maximum(0,start-self.launch_time+rs[:,None]*ramp_s))
        throttle = rm/ramp_limit
        valid &= np.all(np.abs(np.diff(throttle,axis=1))/(rs[:,None]/8)<=self.control['throttle_rate_limit'],axis=1)
        return valid

    def _activate(self,p,v,a,ad,duration,ramp,index,now):
        first,second = ramp_reference(p,v,a,ad,duration,ramp)
        self.active = (now,float(duration),float(ramp),first,second)
        self.plan,self.plan_start,self.plan_duration = first,now,float(ramp)
        if index!=self.target_index:
            self.target_events.append((now,int(index)))
        self.target_index = int(index)
        self.diagnostics['plans'] += 1

    def _make_plan(self,observation,position,velocity,now):
        states = np.asarray(observation['balloon_states'],dtype=float)
        ids = np.flatnonzero(np.asarray(observation['balloon_status']).reshape(-1)==1)
        drift = states[:,3:6]
        force = self.acceleration-G-self.disturbance
        limit = .98*self.available_acceleration(now)
        if limit>0 and np.linalg.norm(force)>limit:
            self.acceleration = G+self.disturbance+force*limit/np.linalg.norm(force)
        if self.target_index in ids and self.active is not None:
            start,duration,_,_,_ = self.active
            remaining = start+duration-now
            if remaining>.12:
                ramps = np.minimum(self.ramp_options,remaining)
                ts = np.full(len(ramps),remaining)
                target = states[self.target_index,:3]+drift[self.target_index]*remaining
                ad,_ = ramp_solution(position,velocity,self.acceleration,target,ts,ramps)
                valid = self._valid(position,velocity,self.acceleration,ad,ts,ramps,now)
                if np.any(valid):
                    k = int(np.flatnonzero(valid)[0])
                    self._activate(position,velocity,self.acceleration,ad[k],remaining,ramps[k],self.target_index,now)
                return
            self.failed_until[self.target_index] = now+1.
        remaining = self.launch_time+self.burn_time-now-1e-4
        beam = [(now,position,velocity,self.acceleration,(),None)]
        best = None
        self.diagnostics['beam_searches'] += 1
        for depth in range(self.search_depth):
            children = []
            for start,p,v,a,route,first in beam:
                candidates = [int(i) for i in ids if i not in route and self.failed_until.get(int(i),0)<=now]
                if not candidates:
                    continue
                projected = states[:,:3]+drift*(start-now)
                estimates = [np.linalg.norm(projected[candidates]+drift[candidates]*t-p-v*t,axis=1)/t**2+.1*t for t in (1.,2.,4.,7.)]
                selected = np.asarray(candidates)[np.argsort(np.min(estimates,axis=0))[:self.branch_targets]]
                ts = np.arange(.5,min(self.leg_horizon,remaining-(start-now))+.001,self.time_grid)
                if not len(ts):
                    continue
                ii,tt,rr = np.meshgrid(selected,ts,self.ramp_options,indexing='ij')
                ii,tt,rr = ii.ravel(),tt.ravel(),rr.ravel()
                keep = rr<=tt
                ii,tt,rr = ii[keep],tt[keep],rr[keep]
                targets = projected[ii]+drift[ii]*tt[:,None]
                ad,ev = ramp_solution(p,v,a,targets,tt,rr)
                valid = self._valid(p,v,a,ad,tt,rr,start)
                self.diagnostics['beam_candidates'] += len(tt)
                self.diagnostics['beam_feasible'] += int(valid.sum())
                for k in np.flatnonzero(valid):
                    leg = (ad[k],float(tt[k]),float(rr[k]),int(ii[k])) if first is None else first
                    children.append((start+tt[k],targets[k],ev[k],ad[k],route+(int(ii[k]),),leg))
            if not children:
                break
            children.sort(key=lambda n:n[0])
            beam,bins = [],set()
            for node in children:
                key = (node[4][0],node[4][-1],tuple(np.round(node[2]/5).astype(int)))
                if key in bins:
                    continue
                bins.add(key)
                beam.append(node)
                if len(beam)>=self.beam_width:
                    break
            best = beam[0]
            self.diagnostics['maximum_depth'] = max(self.diagnostics['maximum_depth'],depth+1)
        if best is None:
            self.active = None
            return super()._make_plan(observation,position,velocity,now)
        ad,duration,ramp,index = best[5]
        self.route_events.append([now,list(best[4]),best[0]-now])
        self._activate(position,velocity,self.acceleration,ad,duration,ramp,index,now)

    def get_action(self,observation):
        now = float(observation['simulation_time'])
        if self.active is not None and self.plan is not None:
            start,duration,ramp,first,second = self.active
            if now>=start+ramp:
                self.plan,self.plan_start,self.plan_duration = second,start+ramp,max(duration-ramp,1e-6)
        return super().get_action(observation)
