"""Refine online chain-beam timings within the given physical control limits."""

import numpy as np
from scipy.optimize import minimize
from BalloonPoppingGymEnv.agents.chain_beam_agent import ChainBeamAgent, free_chain
from BalloonPoppingGymEnv.agents.submission_sqp_v2 import batch_samples, G


class ChainSQPAgent(ChainBeamAgent):
    def __init__(self,given_parameters,solver_iterations=30,**kwargs):
        super().__init__(given_parameters,**kwargs)
        self.solver_iterations = int(solver_iterations)
        self.diagnostics.update(chain_refinements=0,chain_refinement_accepted=0)

    def _make_plan(self,observation,position,velocity,now):
        events = len(self.route_events)
        super()._make_plan(observation,position,velocity,now)
        if len(self.route_events)==events or len(self.route)<2:
            return
        states = np.asarray(observation['balloon_states'],dtype=float)
        route = np.array(self.route)
        initial = np.diff(np.r_[now,self.deadlines])
        if np.min(initial)<=.1:
            return
        def curves(ts):
            targets = states[route,:3]+states[route,3:6]*np.cumsum(ts)[:,None]
            return free_chain(position,velocity,self.acceleration,targets,ts,
                              states[route[-1],3:6],self.terminal_weight)
        def margins(ts,samples=25):
            cs = curves(ts)
            p,_,a,j = batch_samples(cs,ts,samples)
            times = now+np.r_[0.,np.cumsum(ts)[:-1]][:,None]+ts[:,None]*np.linspace(0,1,samples)
            limit = self.thrust/np.maximum(self.dry_mass,self.initial_mass-self.mass_flow*np.maximum(0,times-self.launch_time))
            f = a-G-self.disturbance
            mag = np.linalg.norm(f,axis=2)
            axis = f/np.maximum(mag[:,:,None],1e-9)
            rates = np.linalg.norm(j-axis*np.sum(axis*j,axis=2)[:,:,None],axis=2)/np.maximum(mag,1e-9)
            throttle = mag/limit
            change = np.diff(throttle,axis=1)/np.diff(times,axis=1)
            return np.r_[((limit-mag)/12).ravel(),((mag-.5)/12).ravel(),
                          (axis[:,:,2]-np.cos(self.max_tilt)).ravel(),
                          (self.max_axis_rate**2-rates**2).ravel(),
                          ((p[:,:,2]-self.elevation+.05)/10).ravel(),
                          (self.control['throttle_rate_limit']**2-change**2).ravel(),
                          self.launch_time+self.burn_time-now-sum(ts)-1e-4]
        result = minimize(lambda x:float(np.sum(x)),initial,method='SLSQP',
                          bounds=[(max(.3,t-1.5),t+1.) for t in initial],
                          constraints={'type':'ineq','fun':margins},
                          options={'maxiter':self.solver_iterations,'ftol':1e-6})
        self.diagnostics['chain_refinements'] += 1
        if not np.all(np.isfinite(result.x)):
            return
        for blend in (1.,.98,.95,.9,.8,.6,.4,.2):
            durations = initial+blend*(result.x-initial)
            if sum(durations)>=sum(initial) or np.min(margins(durations,65))<-1e-6:
                continue
            cs = curves(durations)
            _,vs,acs,_ = batch_samples(cs,durations)
            self.deadlines = list(now+np.cumsum(durations))
            self.ends_v,self.ends_a = list(vs[:,-1]),list(acs[:,-1])
            self.plan,self.plan_start,self.plan_duration = cs[0],now,float(durations[0])
            self.diagnostics['chain_refinement_accepted'] += 1
            return
