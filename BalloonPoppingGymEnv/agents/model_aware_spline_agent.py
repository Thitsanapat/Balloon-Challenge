"""Two-intercept guidance with an online reduced ActiveRocketPy force model.

Uses accelerometer specific force, propagated attitude, applied TVC/throttle,
and observed balloon drift. It never reads the simulator's private flight state.
"""

import numpy as np

from BalloonPoppingGymEnv.agents.multi_target_agent import (
    _multiply_quaternions, _rotate_body_to_world,
)
from BalloonPoppingGymEnv.agents.physics_guidance_agent import G, allocate_acceleration, sample_curve
from BalloonPoppingGymEnv.agents.spline_route_agent import SplineRouteAgent


def tvc_thrust_axis(tvc_degrees):
    """ActiveRocketPy dual-axis TVC force direction in body coordinates."""
    angles = np.radians(np.asarray(tvc_degrees,dtype=float))
    first = -np.sin(angles[1])
    second = np.sin(angles[0])
    third = np.sqrt(max(0.,1.-first**2-second**2))
    return np.array([first,second,third])


def quadratic_drag_gain(disturbance,air_velocity,upper=0.005):
    """Least-squares scalar k for disturbance approximately -k*|v_air|*v_air."""
    relative = np.asarray(air_velocity,dtype=float)
    speed = np.linalg.norm(relative)
    if speed<2.:
        return None
    estimate = -float(np.dot(disturbance,relative))/speed**3
    return float(np.clip(estimate,0.,upper))


class ModelAwareSplineAgent(SplineRouteAgent):
    def __init__(self,given_parameters,force_filter=.08,drag_filter=2.,
                 bias_filter=1.,max_drag_gain=.005,**kwargs):
        super().__init__(given_parameters,**kwargs)
        self.force_filter=float(force_filter)
        self.drag_filter=float(drag_filter)
        self.bias_filter=float(bias_filter)
        self.max_drag_gain=float(max_drag_gain)
        if min(self.force_filter,self.drag_filter,self.bias_filter)<=0 or self.max_drag_gain<0:
            raise ValueError("Invalid model filter settings")
        self.wind=np.zeros(3)
        self.drag_gain=0.
        self.model_bias=np.zeros(3)
        self.model_ready=False
        self.diagnostics.update(force_model_steps=0,drag_gain=0.,wind=[0.,0.,0.],
                                model_bias=[0.,0.,0.],specific_force_error_sum=0.)

    def _observed_wind(self,observation):
        states=np.asarray(observation["balloon_states"],dtype=float)
        released=np.flatnonzero(np.asarray(observation["balloon_status"]).reshape(-1)==1)
        moving=released[np.linalg.norm(states[released,3:5],axis=1)>.25]
        if moving.size:
            # Balloon vertical velocity contains buoyant rise, not vertical wind.
            target=np.r_[np.median(states[moving,3:5],axis=0),0.]
            self.wind += self.dt/(1.+self.dt)*(target-self.wind)
        return self.wind

    def _drag(self,velocity):
        relative=np.asarray(velocity,dtype=float)-self.wind
        return -self.drag_gain*np.linalg.norm(relative)*relative

    def _disturbance_at(self,velocity):
        return self.model_bias+self._drag(velocity)

    def _update_force_model(self,observation,velocity,available):
        specific_body=np.asarray(observation["rocket_sensors"][3:6],dtype=float)
        measured_specific=_rotate_body_to_world(self.quaternion,specific_body)
        thrust_body=tvc_thrust_axis(self.previous_tvc)
        nominal=_rotate_body_to_world(self.quaternion,thrust_body)*available*self.previous_throttle
        observed=measured_specific-nominal
        relative=velocity-self.wind
        estimate=quadratic_drag_gain(observed-self.model_bias,relative,self.max_drag_gain)
        if estimate is not None:
            self.drag_gain += self.dt/(self.drag_filter+self.dt)*(estimate-self.drag_gain)
        residual=observed-self._drag(velocity)
        self.model_bias += self.dt/(self.bias_filter+self.dt)*(np.clip(residual,-3.,3.)-self.model_bias)
        self.acceleration += self.dt/(self.force_filter+self.dt)*(measured_specific+G-self.acceleration)
        predicted=nominal+self._disturbance_at(velocity)
        self.diagnostics["force_model_steps"]+=1
        self.diagnostics["specific_force_error_sum"]+=float(np.linalg.norm(predicted-measured_specific))
        self.diagnostics.update(drag_gain=float(self.drag_gain),wind=self.wind.tolist(),
                                model_bias=self.model_bias.tolist())
        self.disturbance=self._disturbance_at(velocity)
        self.model_ready=True

    def _feasible(self,c,duration,now):
        times=np.linspace(0.,duration,33)
        p,v,a,jerk=sample_curve(c,duration,times)
        disturbance=np.array([self._disturbance_at(speed) for speed in v])
        thrust=a-G-disturbance
        magnitude=np.linalg.norm(thrust,axis=1)
        limits=np.array([self.available_acceleration(now+t) for t in times])
        if np.any(magnitude>limits) or np.any(magnitude<.5):
            return False
        if np.any(thrust[:,2]<magnitude*np.cos(self.max_tilt)):
            return False
        axes=thrust/magnitude[:,None]
        axis_rate=np.linalg.norm(jerk-axes*np.sum(axes*jerk,axis=1)[:,None],axis=1)/magnitude
        if np.any(axis_rate>self.max_axis_rate):
            return False
        throttle=magnitude/np.maximum(limits,1e-9)
        if np.any(np.abs(np.diff(throttle)/np.diff(times))>self.control["throttle_rate_limit"]):
            return False
        return bool(np.all(p[:,2]>=self.elevation-.05))

    def get_action(self,observation):
        now=float(observation["simulation_time"])
        if not self.launched and now>=self.launch_time:
            self.launched=True
            self.launch_time=now+self.dt
        sensors=np.asarray(observation["rocket_sensors"],dtype=float)
        tvc,roll,throttle=np.zeros(2),0.,1.
        if self.launched and np.all(np.isfinite(sensors)):
            gyro=sensors[:3]
            increment=(gyro+self.previous_gyro)*self.dt/2
            angle=np.linalg.norm(increment)
            if angle>1e-12:
                dq=np.r_[np.cos(angle/2),np.sin(angle/2)*increment/angle]
                self.quaternion=_multiply_quaternions(self.quaternion,dq)
                self.quaternion/=np.linalg.norm(self.quaternion)
            self.previous_gyro=gyro.copy()
            position,velocity=sensors[6:9],sensors[9:12]
            available=self.available_acceleration(now)
            self._observed_wind(observation)
            self._update_force_model(observation,velocity,available)
            self.previous_velocity=velocity.copy()
            status=np.asarray(observation["balloon_status"]).reshape(-1)
            if self.target_index is not None and status[self.target_index]!=1:
                self.target_index,self.plan,self.next_plan=None,None,now
            if now>=self.next_plan:
                self._make_plan(observation,position,velocity,now)
                self.next_plan=now+self.replan_interval
            if self.plan is not None:
                elapsed=np.clip(now-self.plan_start,0,self.plan_duration)
                p,v,a,jerk=sample_curve(self.plan,self.plan_duration,elapsed)
                wn=self.tracking_frequency
                acceleration=a+wn**2*(p-position)+2*wn*(v-velocity)
                self.diagnostics["position_error_sum"]+=float(np.linalg.norm(p-position))
                self.diagnostics["tracking_steps"]+=1
            else:
                acceleration=np.array([0.,0.,max(0.,-velocity[2])])
                jerk=np.zeros(3)
            requested=acceleration-G-self._disturbance_at(velocity)
            allocated=allocate_acceleration(requested,available,self.max_tilt)
            if np.linalg.norm(allocated-requested)>.1:
                self.diagnostics["saturated_steps"]+=1
            magnitude=np.linalg.norm(allocated)
            axis=allocated/max(magnitude,1e-9) if magnitude>1e-9 else np.array([0.,0.,1.])
            self.last_desired_axis=axis.copy()
            tvc,roll,throttle=self._attitude_action(axis,jerk,magnitude,gyro,available)
        return {"launch":self.launched,"launch_inclination_heading":self.launch_attitude.copy(),
                "tvc":tvc,"roll":roll,"throttle":throttle}
