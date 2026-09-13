"""Replay controls recovered by ``scripts/infer_replay_controls.py``.

External control histories are offline diagnostics and are deliberately marked
as non-submission configurations.  The experiment establishes how much error
comes from control reconstruction before distilling it into an observation-only
policy.
"""

import numpy as np

from BalloonPoppingGymEnv.agents.base_agent import BaseAgent
from BalloonPoppingGymEnv.agents.multi_target_agent import (
    _multiply_quaternions,
    _rotate_body_to_world,
    _rotate_world_to_body,
)
from BalloonPoppingGymEnv.envs.balloon_world import get_initial_attitude


class InferredControlDiagnosticAgent(BaseAgent):
    def __init__(self, given_parameters, control_path, position_gain=.25,
                 velocity_gain=1., attitude_gain=3., rate_gain=10.,
                 acceleration_scale=12., detour_target=None,
                 detour_start=None, detour_peak=None, detour_end=None,
                 detour_scale=1.):
        super().__init__(given_parameters)
        with np.load(control_path) as data:
            self.times = np.asarray(data['times'])
            self.states = np.asarray(data['states'])
            self.controls = np.asarray(data['controls'])
            self.launch_time = float(data['launch_time'])
            self.launch_attitude = np.asarray(data['launch_attitude'])
        self.launched = False
        self.dt = float(given_parameters['simulation']['time_step'])
        self.position_gain = float(position_gain)
        self.velocity_gain = float(velocity_gain)
        self.attitude_gain = float(attitude_gain)
        self.rate_gain = float(rate_gain)
        self.acceleration_scale = float(acceleration_scale)
        self.detour_target = None if detour_target is None else int(detour_target)
        self.detour_start = None if detour_start is None else float(detour_start)
        self.detour_peak = None if detour_peak is None else float(detour_peak)
        self.detour_end = None if detour_end is None else float(detour_end)
        self.detour_scale = float(detour_scale)
        if self.detour_target is not None and not (
            self.detour_start < self.detour_peak < self.detour_end
        ):
            raise ValueError('detour times must satisfy start < peak < end')
        self.quaternion = np.asarray(get_initial_attitude(*self.launch_attitude))
        self.previous_gyro = np.zeros(3)
        self.previous_tvc = np.zeros(2)
        self.previous_roll = 0.
        control = given_parameters['rocket']['control']
        self.max_gimbal = float(control['max_gimbal_angle'])
        self.gimbal_step = float(control['gimbal_rate_limit'])*self.dt
        self.max_roll = float(control['max_roll_torque'])
        self.roll_step = float(control['torque_rate_limit'])*self.dt

    def get_action(self, observation):
        now = float(observation['simulation_time'])
        if not self.launched and now >= self.launch_time:
            self.launched = True
        tvc = np.zeros(2)
        throttle, roll = 1., 0.
        if self.launched and now >= self.times[0]:
            index = int(np.clip(np.searchsorted(self.times, now), 0, len(self.times)-1))
            feedforward = self.controls[index]
            tvc = feedforward[:2].copy()
            throttle = float(feedforward[2])
            roll = float(feedforward[3])
            sensors = np.asarray(observation['rocket_sensors'], dtype=float)
            if np.all(np.isfinite(sensors)):
                gyro = sensors[:3]
                increment = (gyro+self.previous_gyro)*self.dt/2.
                angle = np.linalg.norm(increment)
                if angle > 1e-12:
                    dq = np.r_[np.cos(angle/2.), np.sin(angle/2.)*increment/angle]
                    self.quaternion = _multiply_quaternions(self.quaternion, dq)
                    self.quaternion /= np.linalg.norm(self.quaternion)
                self.previous_gyro = gyro.copy()
                reference_position = self.states[index, :3].copy()
                reference_velocity = self.states[index, 3:6].copy()
                if self.detour_target is not None and self.detour_start < now < self.detour_end:
                    if now <= self.detour_peak:
                        phase = (now-self.detour_start)/(self.detour_peak-self.detour_start)
                        phase_rate = 1./(self.detour_peak-self.detour_start)
                    else:
                        phase = (self.detour_end-now)/(self.detour_end-self.detour_peak)
                        phase_rate = -1./(self.detour_end-self.detour_peak)
                    blend = phase*phase*(3.-2.*phase)
                    blend_rate = (6.*phase-6.*phase*phase)*phase_rate
                    target_state = np.asarray(
                        observation['balloon_states'][self.detour_target], dtype=float,
                    )
                    target_position = target_state[:3]
                    delta = target_position-reference_position
                    reference_velocity += self.detour_scale*(
                        blend*(target_state[3:6]-reference_velocity)+blend_rate*delta
                    )
                    reference_position += (
                        self.detour_scale*blend*delta
                    )
                position_error = reference_position-sensors[6:9]
                velocity_error = reference_velocity-sensors[9:12]
                correction = (self.position_gain*position_error
                              + self.velocity_gain*velocity_error)
                reference_axis = _rotate_body_to_world(
                    self.states[index, 6:10], [0., 0., 1.],
                )
                desired = reference_axis+correction/self.acceleration_scale
                desired /= max(np.linalg.norm(desired), 1e-9)
                body_axis = _rotate_body_to_world(
                    self.quaternion, [0., 0., 1.],
                )
                axis_error = _rotate_world_to_body(
                    self.quaternion, np.cross(body_axis, desired),
                )
                desired_rate = (self.states[index, 10:12]
                                + self.attitude_gain*axis_error[:2])
                tvc += self.rate_gain*(desired_rate-gyro[:2])
                tvc = np.clip(tvc, -self.max_gimbal, self.max_gimbal)
                tvc = self.previous_tvc+np.clip(
                    tvc-self.previous_tvc, -self.gimbal_step, self.gimbal_step,
                )
                throttle += np.dot(correction, reference_axis)/self.acceleration_scale
                throttle = float(np.clip(throttle, 0., 1.))
                roll += 5.*(self.states[index, 12]-gyro[2])
                roll = float(np.clip(roll, -self.max_roll, self.max_roll))
                roll = self.previous_roll+float(np.clip(
                    roll-self.previous_roll, -self.roll_step, self.roll_step,
                ))
                self.previous_tvc = tvc.copy()
                self.previous_roll = roll
        return {
            'launch': self.launched,
            'launch_inclination_heading': self.launch_attitude.copy(),
            'tvc': tvc,
            'roll': roll,
            'throttle': throttle,
        }
