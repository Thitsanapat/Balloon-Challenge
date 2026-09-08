"""Prefer dense, ascending intercepts along the observed cloud's inclined core.

This changes target ranking, not the official balloon distribution or physics.
The approximately quarter-area core is a soft preference, never a fabricated hit.
"""

import numpy as np

from BalloonPoppingGymEnv.agents.physics_guidance_agent import PhysicsGuidanceAgent
from BalloonPoppingGymEnv.agents.spline_route_agent import SplineRouteAgent, two_intercept_spline


def cloud_geometry(points, core_fraction=0.5, density_scale=30.):
    """Fit horizontal displacement versus height; describe an upward cloud axis.

    Half the reference radius gives one quarter of its circular cross-section.
    The reference is the observed 80th-percentile radius, not a board boundary.
    """
    points = np.asarray(points,dtype=float)
    if points.ndim!=2 or points.shape[1]!=3 or not len(points) or not np.isfinite(points).all():
        raise ValueError("Finite nonempty N by 3 points required")
    center = np.mean(points,axis=0)
    centered = points-center
    vertical = centered[:,2]
    variance = float(vertical@vertical)
    slope = vertical@centered[:,:2]/variance if variance>1e-8 else np.zeros(2)
    axis = np.r_[slope,1.]
    axis /= np.linalg.norm(axis)
    progress = centered@axis
    residual = centered-progress[:,None]*axis
    radial = np.linalg.norm(residual,axis=1)
    radius = max(5.,float(core_fraction)*float(np.quantile(radial,.8)))
    deltas = points[:,None,:]-points[None,:,:]
    squared = np.sum(deltas*deltas,axis=2)
    density = np.sum(np.exp(-squared/(2*density_scale**2)),axis=1)-1.
    return center,axis,progress,radial,radius,density


class CloudCorridorAgent(SplineRouteAgent):
    def __init__(self,given_parameters,core_fraction=.5,density_scale=30.,
                 radial_weight=2.,density_weight=4.,descent_weight=2.,**kwargs):
        super().__init__(given_parameters,**kwargs)
        if not 0<core_fraction<=1 or density_scale<=0 or min(radial_weight,density_weight,descent_weight)<0:
            raise ValueError("Invalid corridor settings")
        self.core_fraction,self.density_scale = float(core_fraction),float(density_scale)
        self.radial_weight,self.density_weight,self.descent_weight = float(radial_weight),float(density_weight),float(descent_weight)
        self.diagnostics.update(corridor_plans=0,cloud_tilt_degrees=0.,core_radius=0.,
                                released_count=0,core_count=0,max_neighbor_density=0.)

    def _make_plan(self,observation,position,velocity,now):
        states = np.asarray(observation["balloon_states"],dtype=float)
        released = np.flatnonzero(np.asarray(observation["balloon_status"]).reshape(-1)==1)
        if released.size<3:
            super()._make_plan(observation,position,velocity,now)
            return
        center,axis,progress,radial,core_radius,density = cloud_geometry(
            states[released,:3],self.core_fraction,self.density_scale)
        penalties = np.zeros(len(states))
        penalties[released] = (self.radial_weight*np.maximum(0.,radial/core_radius-1.)
                               +self.density_weight/(1.+density))
        along = (states[:,:3]-center)@axis
        rocket_progress = float((position-center)@axis)
        self.diagnostics.update(cloud_tilt_degrees=float(np.degrees(np.arccos(np.clip(axis[2],-1,1)))),
                                core_radius=float(core_radius),released_count=int(released.size),
                                core_count=int(np.count_nonzero(radial<=core_radius)),
                                max_neighbor_density=float(np.max(density)))
        moving = released[np.linalg.norm(states[released,3:6],axis=1)>.5]
        drift = states[:,3:6].copy()
        if moving.size:
            drift[np.linalg.norm(drift,axis=1)<=.5] = np.median(drift[moving],axis=0)
        candidates = [int(i) for i in released if self.failed_until.get(int(i),0)<=now]
        locked = self.target_index in candidates
        if locked:
            candidates = [self.target_index]
            eta = max(.4,self.plan_start+self.plan_duration-now)
            first_times = np.array([eta,eta+.8,eta+1.6])
        else:
            candidates.sort(key=lambda i: np.linalg.norm(states[i,:3]-position)/10.+penalties[i]
                            +max(0.,along[i]-rocket_progress)/30.)
            candidates = candidates[:self.first_candidates]
            first_times = np.arange(2.,self.horizon+.01,self.duration_step)
        remaining = self.burn_time-(now-self.launch_time)-1e-5
        best = None
        for first_time in first_times:
            if first_time>=remaining or (best is not None and first_time+1.5>=best[0]):
                continue
            predicted = states[:,:3]+drift*first_time
            for first in candidates:
                others = [int(j) for j in released if j!=first]
                def descent(second):
                    return self.descent_weight*max(0.,along[first]-along[second])/30.
                others.sort(key=lambda j: np.linalg.norm(predicted[j]-predicted[first])/10.
                            +penalties[j]+descent(j))
                for second in others[:self.next_candidates]:
                    extra = .5*(penalties[first]+penalties[second])+descent(second)
                    for second_time in np.arange(1.5,self.second_horizon+.01,self.duration_step):
                        total = first_time+second_time
                        cost = total+extra
                        if total>remaining or (best is not None and cost>=best[0]):
                            break
                        self.diagnostics["pair_candidates"] += 1
                        c1,c2 = two_intercept_spline(position,velocity,self.acceleration,
                            predicted[first],states[second,:3]+drift[second]*total,
                            drift[second],first_time,second_time)
                        if self._feasible(c1,first_time,now) and self._feasible(c2,second_time,now+first_time):
                            self.diagnostics["feasible_pairs"] += 1
                            best = cost,total,first,second,first_time,c1
                            break
        if best is None:
            PhysicsGuidanceAgent._make_plan(self,observation,position,velocity,now)
            self.diagnostics["fallback_plans"] += 1
            return
        _,total,first,second,duration,curve = best
        if first!=self.target_index:
            self.target_events.append((now,first))
        self.route_events.append([float(now),first,second,float(duration),float(total)])
        self.target_index = first
        self.plan,self.plan_start,self.plan_duration = curve,now,float(duration)
        self.diagnostics["plans"] += 1
        self.diagnostics["pair_plans"] += 1
        self.diagnostics["corridor_plans"] += 1
