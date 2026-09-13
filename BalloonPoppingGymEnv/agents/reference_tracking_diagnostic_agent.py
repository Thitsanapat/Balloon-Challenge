"""Diagnostic tracker for studying a public replay's feasible GNC envelope.

This agent intentionally reads an external replay and is therefore not a
submission agent.  It is a system-identification tool: if feedback cannot track
an already-flown state history in the same simulator, route search is not the
current bottleneck.
"""

import json
from pathlib import Path

import numpy as np

from BalloonPoppingGymEnv.agents.multi_target_agent import (
    _multiply_quaternions,
    _rotate_body_to_world,
    _rotate_world_to_body,
)
from BalloonPoppingGymEnv.agents.physics_guidance_agent import PhysicsGuidanceAgent
from BalloonPoppingGymEnv.envs.balloon_world import get_initial_attitude


class ReferenceTrackingDiagnosticAgent(PhysicsGuidanceAgent):
    def __init__(self, given_parameters, reference_path,
                 launch_attitude, position_gain=.12, velocity_gain=.7,
                 integral_gain=0., integral_limit=20.,
                 correction_times=None, correction_vectors=None,
                 correction_sigma=.6, correction_scale=1.,
                 attitude_gain=3., rate_gain=10., control_mode='rate',
                 angular_rate_gain=8., angular_acceleration_gain=16.,
                 reference_attitude_lead=0.,
                 **kwargs):
        super().__init__(given_parameters, **kwargs)
        payload = json.loads(Path(reference_path).read_text(encoding='utf-8'))
        records = payload['trajectories']
        valid = [r for r in records if r['rocket_states'][0] is not None]
        self.reference_time = np.asarray([r['time'] for r in valid], dtype=float)
        self.reference_state = np.asarray([r['rocket_states'] for r in valid], dtype=float)
        self.reference_axis = np.asarray([
            _rotate_body_to_world(state[6:10], [0., 0., 1.])
            for state in self.reference_state
        ])
        self.reference_axis_rate = np.gradient(
            self.reference_axis, self.reference_time, axis=0,
        )
        self.reference_omega_world = np.asarray([
            _rotate_body_to_world(state[6:10], state[10:13])
            for state in self.reference_state
        ])
        self.reference_angular_acceleration = np.gradient(
            self.reference_state[:, 10:13], self.reference_time, axis=0,
        )
        self.launch_attitude = np.asarray(launch_attitude, dtype=float)
        self.quaternion = np.asarray(get_initial_attitude(*self.launch_attitude))
        self.position_gain = float(position_gain)
        self.velocity_gain = float(velocity_gain)
        self.integral_gain = float(integral_gain)
        self.integral_limit = float(integral_limit)
        self.position_integral = np.zeros(3)
        self.correction_times = np.asarray(correction_times or [], dtype=float)
        self.correction_vectors = np.asarray(correction_vectors or [], dtype=float)
        if self.correction_times.size:
            if self.correction_vectors.shape != (len(self.correction_times), 3):
                raise ValueError('correction_vectors must contain one vector per time')
        else:
            self.correction_vectors = np.empty((0, 3))
        self.correction_sigma = float(correction_sigma)
        self.correction_scale = float(correction_scale)
        self.attitude_gain = float(attitude_gain)
        self.rate_gain = float(rate_gain)
        self.control_mode = str(control_mode)
        if self.control_mode not in {'rate', 'computed_torque'}:
            raise ValueError("control_mode must be 'rate' or 'computed_torque'")
        self.angular_rate_gain = float(angular_rate_gain)
        self.angular_acceleration_gain = float(angular_acceleration_gain)
        self.reference_attitude_lead = float(reference_attitude_lead)
        self.diagnostics.update(
            reference_error_sum=0., reference_error_vector=[0., 0., 0.],
            reference_velocity_error=[0., 0., 0.], axis_error_sum=0.,
            axis_error_max=0., reference_steps=0,
        )

    def _reference(self, now):
        index = int(np.clip(
            np.searchsorted(self.reference_time, now),
            1,
            len(self.reference_time)-1,
        ))
        before, after = index-1, index
        span = self.reference_time[after]-self.reference_time[before]
        fraction = np.clip((now-self.reference_time[before])/span, 0., 1.)
        state = ((1.-fraction)*self.reference_state[before]
                 + fraction*self.reference_state[after])
        axis = ((1.-fraction)*self.reference_axis[before]
                + fraction*self.reference_axis[after])
        axis /= max(np.linalg.norm(axis), 1e-9)
        omega_world = ((1.-fraction)*self.reference_omega_world[before]
                       + fraction*self.reference_omega_world[after])
        angular_acceleration = (
            (1.-fraction)*self.reference_angular_acceleration[before]
            + fraction*self.reference_angular_acceleration[after]
        )
        return state, axis, omega_world, angular_acceleration

    def get_action(self, observation):
        now = float(observation['simulation_time'])
        if not self.launched and now >= self.launch_time:
            self.launched = True
            self.launch_time = now+self.dt
        tvc, roll = np.zeros(2), 0.
        sensors = np.asarray(observation['rocket_sensors'], dtype=float)
        if self.launched and np.all(np.isfinite(sensors)):
            gyro = sensors[:3]
            increment = (gyro+self.previous_gyro)*self.dt/2.
            angle = np.linalg.norm(increment)
            if angle > 1e-12:
                dq = np.r_[np.cos(angle/2.), np.sin(angle/2.)*increment/angle]
                self.quaternion = _multiply_quaternions(self.quaternion, dq)
                self.quaternion /= np.linalg.norm(self.quaternion)
            self.previous_gyro = gyro.copy()
            position, velocity = sensors[6:9], sensors[9:12]
            reference, _, _, _ = self._reference(now)
            _, feedforward_axis, omega_world, angular_acceleration = self._reference(
                now+self.reference_attitude_lead
            )
            if self.correction_times.size:
                weights = np.exp(
                    -.5*((now-self.correction_times)/self.correction_sigma)**2
                )
                reference[:3] += self.correction_scale*(weights @ self.correction_vectors)
            available = self.available_acceleration(now)
            position_error = reference[:3]-position
            self.position_integral += position_error*self.dt
            norm = np.linalg.norm(self.position_integral)
            if norm > self.integral_limit:
                self.position_integral *= self.integral_limit/norm
            requested = (available*feedforward_axis
                         + self.position_gain*position_error
                         + self.velocity_gain*(reference[3:6]-velocity)
                         + self.integral_gain*self.position_integral)
            desired_axis = requested/max(np.linalg.norm(requested), 1e-9)
            self.last_desired_axis = desired_axis.copy()
            body_axis = _rotate_body_to_world(self.quaternion, [0., 0., 1.])
            axis_error = float(np.arccos(np.clip(np.dot(body_axis, feedforward_axis), -1., 1.)))
            attitude_error = _rotate_world_to_body(
                self.quaternion, np.cross(body_axis, desired_axis),
            )
            omega_body = _rotate_world_to_body(self.quaternion, omega_world)
            desired_rate = np.clip(
                omega_body[:2]+self.attitude_gain*attitude_error[:2],
                -self.max_axis_rate,
                self.max_axis_rate,
            )
            if self.control_mode == 'computed_torque':
                requested_angular_acceleration = (
                    angular_acceleration[:2]
                    + self.angular_rate_gain*(desired_rate-gyro[:2])
                    + self.angular_acceleration_gain*attitude_error[:2]
                )
                torque = self.inertia[:2]*requested_angular_acceleration
                tvc = np.degrees(np.arcsin(np.clip(
                    torque/(self.thrust*self.lever), -1., 1.,
                )))
            else:
                tvc = self.rate_gain*(desired_rate-gyro[:2])
            tvc = np.clip(
                tvc,
                -self.control['max_gimbal_angle'],
                self.control['max_gimbal_angle'],
            )
            max_change = self.control['gimbal_rate_limit']*self.dt
            tvc = self.previous_tvc+np.clip(
                tvc-self.previous_tvc, -max_change, max_change,
            )
            roll = float(np.clip(
                -20.*gyro[2],
                -self.control['max_roll_torque'],
                self.control['max_roll_torque'],
            ))
            self.previous_tvc = tvc.copy()
            self.diagnostics['reference_error_sum'] += float(
                np.linalg.norm(reference[:3]-position)
            )
            self.diagnostics['reference_error_vector'] = (
                np.asarray(self.diagnostics['reference_error_vector'])+position_error
            ).tolist()
            self.diagnostics['reference_velocity_error'] = (
                np.asarray(self.diagnostics['reference_velocity_error'])
                + reference[3:6]-velocity
            ).tolist()
            self.diagnostics['axis_error_sum'] += axis_error
            self.diagnostics['axis_error_max'] = max(
                self.diagnostics['axis_error_max'], axis_error,
            )
            self.diagnostics['reference_steps'] += 1
        return {
            'launch': self.launched,
            'launch_inclination_heading': self.launch_attitude.copy(),
            'tvc': tvc,
            'roll': roll,
            'throttle': 1.,
        }
