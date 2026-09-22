"""Track a trained target schedule as one smooth ascent/descent route.

The schedule is an offline training output. During evaluation every waypoint is
rebuilt from the current observation; no simulator object or future field is
read. Unreleased targets can be predicted when their learned release times are
supplied, which is intentionally seed-specific for the leaderboard round.
"""

import numpy as np

from BalloonPoppingGymEnv.agents.physics_guidance_agent import (
    G, PhysicsGuidanceAgent,
)
from BalloonPoppingGymEnv.agents.multi_target_agent import (
    _rotate_body_to_world, _rotate_world_to_body,
)
from BalloonPoppingGymEnv.agents.beam_intercept_agent import flythrough_curves
from BalloonPoppingGymEnv.agents.three_intercept_agent import intercept_chain
from BalloonPoppingGymEnv.agents.spline_route_agent import boundary_curve
from BalloonPoppingGymEnv.envs.balloon_world import get_initial_attitude


class ScheduledAscentAgent(PhysicsGuidanceAgent):
    def __init__(self, given_parameters, target_sequence, target_times,
                 release_times=None, route_horizon=5, prediction_speed=6.1,
                 missed_grace=.8, braking_fraction=0., endpoint_velocities=None,
                 guidance_mode='chain', aggressive_rate_control=False,
                 attitude_gain=2.8, rate_gain=10.0, force_full_throttle=False,
                 launch_attitude_override=None, schedule_delay=0.0,
                 waypoint_offset=(0.0, 0.0, 0.0),
                 **kwargs):
        super().__init__(given_parameters, **kwargs)
        self.sequence = [int(index) for index in target_sequence]
        self.deadlines = np.asarray(target_times, dtype=float)
        if len(self.sequence) != len(self.deadlines) or not len(self.sequence):
            raise ValueError('target_sequence and target_times must match and be nonempty')
        if np.any(np.diff(self.deadlines) <= 0):
            raise ValueError('target_times must be strictly increasing')
        self.release_times = (None if release_times is None else
                              np.asarray(release_times, dtype=float))
        if self.release_times is not None and len(self.release_times) != len(self.sequence):
            raise ValueError('release_times must match target_sequence')
        self.route_horizon = int(route_horizon)
        self.prediction_speed = float(prediction_speed)
        self.missed_grace = float(missed_grace)
        self.braking_fraction = float(braking_fraction)
        self.endpoint_velocities = (None if endpoint_velocities is None else
                                    np.asarray(endpoint_velocities, dtype=float))
        if self.endpoint_velocities is not None and self.endpoint_velocities.shape != (len(self.sequence), 3):
            raise ValueError('endpoint_velocities must be one 3-vector per target')
        self.guidance_mode = str(guidance_mode)
        if self.guidance_mode not in {'chain', 'tangent', 'natural', 'zem'}:
            raise ValueError('unknown guidance_mode')
        self.pointer = 0
        self.aggressive_rate_control = bool(aggressive_rate_control)
        self.attitude_gain = float(attitude_gain)
        self.rate_gain = float(rate_gain)
        self.force_full_throttle = bool(force_full_throttle)
        self.launch_attitude_override = (None if launch_attitude_override is None else
                                         np.asarray(launch_attitude_override, dtype=float))
        if self.launch_attitude_override is not None and self.launch_attitude_override.shape != (2,):
            raise ValueError('launch_attitude_override must be [inclination, heading]')
        self.schedule_delay = float(schedule_delay)
        self.waypoint_offset = np.asarray(waypoint_offset, dtype=float)
        if self.waypoint_offset.shape != (3,):
            raise ValueError('waypoint_offset must be an xyz vector')
        self.launch_initialized = False
        self.diagnostics.update(schedule_plans=0, schedule_skips=0)

    def _attitude_action(self, axis, jerk, magnitude, gyro, available):
        """Optionally use the workshop's cascaded attitude/rate controller.

        The base controller converts a desired angular acceleration through an
        approximate inertia model.  That is useful for smooth paths but proved
        too conservative for the short, two-to-three second gaps in a trained
        ascent schedule.  This alternate controller closes the measured body-
        rate loop directly, following the public workshop example.
        """
        if not self.aggressive_rate_control:
            return super()._attitude_action(axis, jerk, magnitude, gyro, available)
        body_axis = _rotate_body_to_world(self.quaternion, [0., 0., 1.])
        error_world = np.cross(body_axis, axis)
        error_body = _rotate_world_to_body(self.quaternion, error_world)
        desired_rate = np.clip(
            self.attitude_gain*error_body[:2],
            -self.max_axis_rate,
            self.max_axis_rate,
        )
        tvc = np.clip(
            self.rate_gain*(desired_rate-gyro[:2]),
            -self.control['max_gimbal_angle'],
            self.control['max_gimbal_angle'],
        )
        # Preserve the actuator rate limits in the command too.  This avoids a
        # one-step discrepancy between our controller state and the actuator.
        max_change = self.control['gimbal_rate_limit']*self.dt
        tvc = self.previous_tvc + np.clip(
            tvc-self.previous_tvc, -max_change, max_change,
        )
        if self.force_full_throttle:
            throttle = 1.0
        else:
            throttle = float(np.clip(
                magnitude/max(available, 1e-9), *self.control['throttle_range'],
            ))
            max_change = self.control['throttle_rate_limit']*self.dt
            throttle = self.previous_throttle+float(np.clip(
                throttle-self.previous_throttle, -max_change, max_change,
            ))
        roll = float(np.clip(
            -20.*gyro[2],
            -self.control['max_roll_torque'],
            self.control['max_roll_torque'],
        ))
        self.previous_tvc = tvc.copy()
        self.previous_throttle = throttle
        return tvc, roll, throttle

    def _shared_drift(self, states, status):
        active = np.flatnonzero((status == 1) & (np.linalg.norm(states[:, 3:6], axis=1) > .5))
        return np.median(states[active, 3:6], axis=0) if active.size else np.array([0., 0., self.prediction_speed])

    def _waypoint(self, states, status, schedule_index, now, arrival, shared):
        index = self.sequence[schedule_index]
        if status[index] == 1:
            return states[index, :3] + states[index, 3:6] * max(0., arrival-now)
        if self.release_times is None:
            return states[index, :3]
        age = max(0., arrival-self.release_times[schedule_index])
        velocity = shared.copy()
        velocity[2] = self.prediction_speed
        return states[index, :3] + velocity*age

    def _advance(self, status, now):
        while self.pointer < len(self.sequence):
            target = self.sequence[self.pointer]
            if status[target] == 2 or now > self.deadlines[self.pointer]+self.schedule_delay+self.missed_grace:
                self.diagnostics['schedule_skips'] += int(status[target] != 2)
                self.pointer += 1
            else:
                break

    def _make_plan(self, observation, position, velocity, now):
        states = np.asarray(observation['balloon_states'], dtype=float)
        status = np.asarray(observation['balloon_status']).reshape(-1)
        self._advance(status, now)
        if self.pointer >= len(self.sequence):
            self.target_index = None
            self.plan = None
            return
        effective_deadlines = self.deadlines+self.schedule_delay
        end = min(len(self.sequence), self.pointer+self.route_horizon)
        schedule_indices = np.arange(self.pointer, end)
        schedule_indices = schedule_indices[effective_deadlines[schedule_indices] > now+.15]
        if not schedule_indices.size:
            return
        shared = self._shared_drift(states, status)
        arrivals = effective_deadlines[schedule_indices]
        waypoints = np.asarray([self._waypoint(states, status, int(k), now, arrival, shared)
                                for k,arrival in zip(schedule_indices,arrivals)])
        waypoints += self.waypoint_offset
        durations = np.diff(np.r_[now, arrivals])
        if self.guidance_mode == 'zem':
            duration = durations[0]
            acceleration = 2.*(waypoints[0]-position-velocity*duration)/duration**2
            curve = np.zeros((6, 3))
            curve[0], curve[1], curve[2] = position, velocity*duration, acceleration*duration**2/2.
        elif self.endpoint_velocities is not None:
            terminal_velocity = self.endpoint_velocities[self.pointer]
            curve = boundary_curve(position, velocity, self.acceleration,
                                   waypoints[0], terminal_velocity, np.zeros(3), durations[0])
        elif self.guidance_mode == 'tangent' and len(durations) >= 2:
            # Carry momentum toward the following balloon instead of arriving
            # with the target's slow drift velocity.
            terminal_velocity = (waypoints[1]-waypoints[0])/durations[1]
            curve = boundary_curve(position, velocity, self.acceleration,
                                   waypoints[0], terminal_velocity, np.zeros(3), durations[0])
        elif self.guidance_mode == 'chain' and len(durations) >= 2:
            final_velocity = shared.copy()
            if end < len(self.sequence):
                following = self._waypoint(states, status, end, now, effective_deadlines[end], shared)
                final_velocity = (following-waypoints[-1])/(effective_deadlines[end]-arrivals[-1])
            curves = intercept_chain(position, velocity, self.acceleration,
                                     waypoints, final_velocity, durations)
            curve = curves[0]
        else:
            curve = flythrough_curves(position, velocity, self.acceleration,
                                      waypoints[:1], states[[self.sequence[self.pointer]], 3:6],
                                      durations[:1], [self.braking_fraction])[0]
        target = self.sequence[self.pointer]
        active_target = target if status[target] == 1 else None
        if active_target is not None and target != self.target_index:
            self.target_events.append((now, target))
        # PhysicsGuidanceAgent clears plans for a non-released target. Keep the
        # schedule curve active while waiting for a learned future release.
        self.target_index = active_target
        self.plan, self.plan_start, self.plan_duration = curve, now, float(durations[0])
        self.diagnostics['plans'] += 1
        self.diagnostics['schedule_plans'] += 1

    def get_action(self, observation):
        now = float(observation['simulation_time'])
        if not self.launch_initialized and now >= self.launch_time:
            states = np.asarray(observation['balloon_states'], dtype=float)
            status = np.asarray(observation['balloon_status']).reshape(-1)
            shared = self._shared_drift(states, status)
            target = (self._waypoint(states, status, 0, now, self.deadlines[0], shared)
                      + self.waypoint_offset)
            duration = max(self.deadlines[0]-now, .5)
            acceleration = 2.*(target-np.array([0., 0., self.elevation]))/duration**2
            axis = acceleration-G
            axis /= max(np.linalg.norm(axis), 1e-9)
            inclination = np.degrees(np.arcsin(np.clip(axis[2], -1., 1.)))
            heading = np.degrees(np.arctan2(axis[0], axis[1])) % 360.
            self.launch_attitude = (np.array([inclination, heading])
                                    if self.launch_attitude_override is None else
                                    self.launch_attitude_override.copy())
            inclination, heading = self.launch_attitude
            self.quaternion = np.asarray(get_initial_attitude(inclination, heading))
            self.launch_initialized = True
        return super().get_action(observation)
