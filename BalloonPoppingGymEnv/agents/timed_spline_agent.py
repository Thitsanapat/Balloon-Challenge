"""Refine two-intercept durations continuously after discrete route selection."""

import numpy as np
from scipy.optimize import minimize

from BalloonPoppingGymEnv.agents.spline_route_agent import SplineRouteAgent, two_intercept_spline
from BalloonPoppingGymEnv.agents.optimized_spiral_agent import OptimizedSpiralAgent


class TimedSplineAgent(SplineRouteAgent):
    def __init__(self,given_parameters,**kwargs):
        super().__init__(given_parameters,**kwargs)
        self.diagnostics.update(timing_calls=0,timing_improvements=0,timing_non_success=0,
                                planned_seconds_saved=0.)

    def _make_plan(self,observation,position,velocity,now):
        old_target = self.target_index
        previous_pairs = self.diagnostics["pair_plans"]
        super()._make_plan(observation,position,velocity,now)
        if self.diagnostics["pair_plans"]==previous_pairs:
            return
        _,first,second,t1,total = self.route_events[-1]
        t2 = total-t1
        states = np.asarray(observation["balloon_states"],dtype=float)
        released = np.flatnonzero(np.asarray(observation["balloon_status"]).reshape(-1)==1)
        moving = released[np.linalg.norm(states[released,3:6],axis=1)>.5]
        drift = states[:,3:6].copy()
        if moving.size:
            drift[np.linalg.norm(drift,axis=1)<=.5] = np.median(drift[moving],axis=0)
        remaining = self.burn_time-(now-self.launch_time)-1e-5

        def curves(times):
            return two_intercept_spline(position,velocity,self.acceleration,
                states[first,:3]+drift[first]*times[0],
                states[second,:3]+drift[second]*sum(times),drift[second],*times)

        def constraints(times):
            c1,c2 = curves(times)
            return np.r_[OptimizedSpiralAgent._constraint_margins(self,c1,times[0],now),
                         OptimizedSpiralAgent._constraint_margins(self,c2,times[1],now+times[0]),
                         remaining-sum(times)]-1e-5

        def objective(times):
            # Do not defer a held first target to improve only a distant second hit.
            delay = 4.*max(0.,times[0]-t1) if first==old_target else 0.
            return float(sum(times)+delay)

        self.diagnostics["timing_calls"] += 1
        result = minimize(objective,[t1,t2],method="SLSQP",
                          bounds=[(max(.2,t1-1.5),t1+.5),(max(.3,t2-1.5),t2+.5)],
                          constraints={"type":"ineq","fun":constraints},
                          options={"maxiter":20,"ftol":1e-5})
        if not result.success:
            self.diagnostics["timing_non_success"] += 1
        if not np.all(np.isfinite(result.x)) or sum(result.x)>remaining or objective(result.x)>=total-1e-6:
            return
        c1,c2 = curves(result.x)
        if self._feasible(c1,result.x[0],now) and self._feasible(c2,result.x[1],now+result.x[0]):
            self.plan = c1
            self.plan_duration = float(result.x[0])
            self.diagnostics["timing_improvements"] += 1
            # Sum of overlapping local forecasts, NOT actual episode time saved.
            self.diagnostics["planned_seconds_saved"] += float(total-sum(result.x))
