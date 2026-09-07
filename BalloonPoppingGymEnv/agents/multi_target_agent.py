"""Closed-loop multi-target guidance for the Balloon Popping Challenge."""

import numpy as np

from BalloonPoppingGymEnv.agents.base_agent import BaseAgent
from BalloonPoppingGymEnv.envs.balloon_world import get_initial_attitude


def _multiply_quaternions(left, right):
    """Hamilton product for quaternions stored as [w, x, y, z]."""
    w1, x1, y1, z1 = left
    w2, x2, y2, z2 = right
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )


def _rotate_body_to_world(quaternion, vector):
    """Rotate a three-vector with q * [0, v] * conjugate(q)."""
    pure = np.concatenate(([0.0], np.asarray(vector, dtype=float)))
    conjugate = quaternion * np.array([1.0, -1.0, -1.0, -1.0])
    return _multiply_quaternions(
        _multiply_quaternions(quaternion, pure), conjugate
    )[1:]


def _rotate_world_to_body(quaternion, vector):
    conjugate = quaternion * np.array([1.0, -1.0, -1.0, -1.0])
    pure = np.concatenate(([0.0], np.asarray(vector, dtype=float)))
    return _multiply_quaternions(
        _multiply_quaternions(conjugate, pure), quaternion
    )[1:]


class MultiTargetGuidanceAgent(BaseAgent):
    """Track released balloons with velocity and attitude feedback."""

    def __init__(
        self,
        given_parameters,
        min_launch_time=4.0,
        launch_inclination=78.0,
        cruise_speed=28.0,
        velocity_gain=0.8,
        attitude_gain=1.6,
        rate_gain=80.0,
        max_body_rate=0.55,
        max_tilt=42.0,
        target_timeout=8.0,
        failed_target_cooldown=4.0,
        initial_coast_time=7.0,
        approach_speed=5.0,
        launch_lead_time=6.0,
        guidance_mode="pursuit",
        turn_penalty=3.0,
        cluster_weight=0.6,
        min_target_hold=0.8,
        passed_margin=8.0,
        initial_target_index=None,
        coast_throttle=1.0,
        terminal_distance=20.0,
        launch_heading_offset=0.0,
        scripted_targets=None,
        terminal_bias=(0.0, 0.0, 0.0),
        terminal_bias_start=0.0,
        handoff_distance=0.0,
        terminal_distance_late=None,
        terminal_switch_time=0.0,
        max_tilt_late=None,
        max_tilt_switch_time=0.0,
        attitude_gain_late=None,
        max_body_rate_late=None,
        attitude_switch_time=0.0,
        scripted_handoff_distances=None,
        guidance_bias=(0.0, 0.0, 0.0),
        guidance_bias_start=0.0,
        guidance_bias_targets=None,
    ):
        super().__init__(given_parameters)
        self.min_launch_time = float(min_launch_time)
        self.launch_inclination = float(launch_inclination)
        self.cruise_speed = float(cruise_speed)
        self.velocity_gain = float(velocity_gain)
        self.attitude_gain = float(attitude_gain)
        self.rate_gain = float(rate_gain)
        self.max_body_rate = float(max_body_rate)
        self.max_tilt = np.radians(float(max_tilt))
        self.target_timeout = float(target_timeout)
        self.failed_target_cooldown = float(failed_target_cooldown)
        self.initial_coast_time = float(initial_coast_time)
        self.approach_speed = float(approach_speed)
        self.launch_lead_time = float(launch_lead_time)
        self.guidance_mode = str(guidance_mode)
        self.turn_penalty = float(turn_penalty)
        self.cluster_weight = float(cluster_weight)
        self.min_target_hold = float(min_target_hold)
        self.passed_margin = float(passed_margin)
        self.initial_target_index = (
            None if initial_target_index is None else int(initial_target_index)
        )
        self.coast_throttle = float(coast_throttle)
        self.terminal_distance = float(terminal_distance)
        self.launch_heading_offset = float(launch_heading_offset)
        self.scripted_targets = (
            [] if scripted_targets is None else [int(index) for index in scripted_targets]
        )
        self.scripted_pointer = 0
        self.terminal_bias = np.asarray(terminal_bias, dtype=float)
        self.terminal_bias_start = float(terminal_bias_start)
        self.handoff_distance = float(handoff_distance)
        self.terminal_distance_late = (
            self.terminal_distance
            if terminal_distance_late is None
            else float(terminal_distance_late)
        )
        self.terminal_switch_time = float(terminal_switch_time)
        self.max_tilt_late = (
            self.max_tilt
            if max_tilt_late is None
            else np.radians(float(max_tilt_late))
        )
        self.max_tilt_switch_time = float(max_tilt_switch_time)
        self.attitude_gain_late = (
            self.attitude_gain
            if attitude_gain_late is None
            else float(attitude_gain_late)
        )
        self.max_body_rate_late = (
            self.max_body_rate
            if max_body_rate_late is None
            else float(max_body_rate_late)
        )
        self.attitude_switch_time = float(attitude_switch_time)
        self.scripted_handoff_distances = (
            []
            if scripted_handoff_distances is None
            else [float(distance) for distance in scripted_handoff_distances]
        )
        self.guidance_bias = np.asarray(guidance_bias, dtype=float)
        self.guidance_bias_start = float(guidance_bias_start)
        self.guidance_bias_targets = (
            None
            if guidance_bias_targets is None
            else {int(index) for index in guidance_bias_targets}
        )

        self.dt = 1.0 / float(given_parameters["rocket"]["sensors"]["sampling_rate"])
        control = given_parameters["rocket"]["control"]
        self.max_gimbal = float(control["max_gimbal_angle"])
        self.max_roll_torque = float(control["max_roll_torque"])
        self.elevation = float(given_parameters["environment"]["elevation"])
        self.balloon_radius = float(given_parameters["balloon"]["radius"])

        self.launched = False
        self.launch_attitude = np.array([90.0, 0.0])
        self.quaternion = np.array([1.0, 0.0, 0.0, 0.0])
        self.launch_axis = np.array([0.0, 0.0, 1.0])
        self.previous_gyro = np.zeros(3)
        self.target_index = None
        self.target_acquired_at = 0.0
        self.minimum_target_range = np.inf
        self.failed_until = {}
        self.target_events = []
        self.last_desired_axis = np.array([0.0, 0.0, 1.0])

    @staticmethod
    def _released_states(observation):
        states = np.asarray(observation["balloon_states"], dtype=float)
        status = np.asarray(observation["balloon_status"], dtype=int).reshape(-1)
        return states, status, np.flatnonzero(status == 1)

    @staticmethod
    def _shared_balloon_drift(states, released):
        velocities = states[released, 3:6]
        moving = np.linalg.norm(velocities, axis=1) > 0.5
        if not np.any(moving):
            return np.zeros(3)
        return np.median(velocities[moving], axis=0)

    def _balloon_velocity(self, state, shared_drift):
        velocity = state[3:6]
        return shared_drift if np.linalg.norm(velocity) <= 0.5 else velocity

    def _rocket_state(self, observation):
        sensors = np.asarray(observation["rocket_sensors"], dtype=float)
        if np.all(np.isfinite(sensors[6:12])):
            return sensors[6:9], sensors[9:12]
        return np.array([0.0, 0.0, self.elevation]), np.zeros(3)

    def _select_target(self, observation, position, velocity):
        states, status, released = self._released_states(observation)
        if self.scripted_targets:
            now = float(observation["simulation_time"])
            while self.scripted_pointer < len(self.scripted_targets):
                scripted = self.scripted_targets[self.scripted_pointer]
                expired = (
                    self.target_index == scripted
                    and now - self.target_acquired_at > self.target_timeout
                )
                handoff = False
                handoff_distance = self.handoff_distance
                if self.scripted_pointer < len(self.scripted_handoff_distances):
                    handoff_distance = self.scripted_handoff_distances[
                        self.scripted_pointer
                    ]
                if status[scripted] == 1 and handoff_distance > 0.0:
                    relative = states[scripted, :3] - position
                    drift = self._shared_balloon_drift(states, released)
                    target_velocity = self._balloon_velocity(states[scripted], drift)
                    handoff = (
                        np.linalg.norm(relative) <= handoff_distance
                        and np.dot(relative, velocity - target_velocity) > 0.0
                    )
                if status[scripted] != 2 and not expired and not handoff:
                    break
                self.scripted_pointer += 1
            if self.scripted_pointer < len(self.scripted_targets):
                scripted = self.scripted_targets[self.scripted_pointer]
                if self.target_index != scripted:
                    self.target_index = scripted
                    self.target_acquired_at = now
                    self.minimum_target_range = np.inf
                    self.target_events.append((now, scripted))
                if status[scripted] == 0:
                    return states[scripted], np.zeros(3)
                drift = self._shared_balloon_drift(states, released)
                return states[scripted], self._balloon_velocity(
                    states[scripted], drift
                )

        if released.size == 0:
            self.target_index = None
            return None, None

        now = float(observation["simulation_time"])
        if self.target_index is not None and status[self.target_index] == 1:
            relative = states[self.target_index, :3] - position
            distance = np.linalg.norm(relative)
            drift = self._shared_balloon_drift(states, released)
            target_velocity = self._balloon_velocity(states[self.target_index], drift)
            closing = np.dot(relative, velocity - target_velocity) > 0.0
            self.minimum_target_range = min(self.minimum_target_range, distance)
            timed_out = now - self.target_acquired_at > self.target_timeout
            passed = (
                now - self.target_acquired_at >= self.min_target_hold
                and not closing
                and distance > self.minimum_target_range + self.passed_margin
            )
            if not timed_out and not passed:
                return states[self.target_index], self._balloon_velocity(
                    states[self.target_index], drift
                )
            self.failed_until[self.target_index] = now + self.failed_target_cooldown

        available = np.asarray(
            [index for index in released if self.failed_until.get(int(index), 0.0) <= now],
            dtype=int,
        )
        if available.size == 0:
            available = released

        drift = self._shared_balloon_drift(states, released)
        candidate_velocities = np.asarray(
            [self._balloon_velocity(states[index], drift) for index in available]
        )
        relative = states[available, :3] - position
        distance = np.linalg.norm(relative, axis=1)
        if np.linalg.norm(velocity) > 3.0:
            direction = velocity / np.linalg.norm(velocity)
            alignment = relative @ direction / np.maximum(distance, 1e-9)
            travel_time = distance / max(self.cruise_speed, 1.0)
            score = travel_time + self.turn_penalty * (1.0 - alignment)
        else:
            travel_time = distance / max(self.cruise_speed, 1.0)
            score = travel_time

        if available.size > 1 and self.cluster_weight > 0.0:
            predicted = states[available, :3] + candidate_velocities * travel_time[:, None]
            pairwise = np.linalg.norm(predicted[:, None, :] - predicted[None, :, :], axis=2)
            neighbors = min(4, available.size - 1)
            local_spacing = np.partition(pairwise, neighbors, axis=1)[:, 1 : neighbors + 1].mean(axis=1)
            score += self.cluster_weight * local_spacing / max(self.cruise_speed, 1.0)

        self.target_index = int(available[np.argmin(score)])
        self.target_acquired_at = now
        self.minimum_target_range = np.inf
        self.target_events.append((now, self.target_index))
        target = states[self.target_index]
        return target, self._balloon_velocity(target, drift)

    def _initial_attitude(self, observation):
        position = np.array([0.0, 0.0, self.elevation])
        states, _, released = self._released_states(observation)
        if self.initial_target_index is not None:
            target = states[self.initial_target_index]
            drift = self._shared_balloon_drift(states, released)
            target_velocity = self._balloon_velocity(target, drift)
            self.target_index = self.initial_target_index
            self.target_acquired_at = float(observation["simulation_time"])
            self.minimum_target_range = np.inf
            self.target_events.append(
                (float(observation["simulation_time"]), self.target_index)
            )
        else:
            target, target_velocity = self._select_target(
                observation, position, np.zeros(3)
            )
        if target is None:
            return np.array([90.0, 0.0])
        predicted = target[:3] + self.launch_lead_time * target_velocity
        delta = predicted - position
        heading = (
            np.degrees(np.arctan2(delta[0], delta[1]))
            + self.launch_heading_offset
        ) % 360.0
        return np.array([self.launch_inclination, heading])

    def _update_attitude(self, gyro):
        increment = 0.5 * (gyro + self.previous_gyro) * self.dt
        angle = np.linalg.norm(increment)
        if angle > 1e-12:
            delta = np.concatenate(
                ([np.cos(angle / 2.0)], np.sin(angle / 2.0) * increment / angle)
            )
            self.quaternion = _multiply_quaternions(self.quaternion, delta)
            self.quaternion /= np.linalg.norm(self.quaternion)
        self.previous_gyro = gyro.copy()

    def _limit_tilt(self, acceleration, now):
        acceleration = np.asarray(acceleration, dtype=float).copy()
        acceleration[2] = max(acceleration[2], 2.0)
        horizontal = np.linalg.norm(acceleration[:2])
        time_since_launch = max(0.0, now - self.min_launch_time)
        tilt_fraction = np.clip(time_since_launch / 8.0, 0.0, 1.0)
        configured_max_tilt = (
            self.max_tilt_late if now >= self.max_tilt_switch_time else self.max_tilt
        )
        allowed_tilt = np.radians(20.0) + tilt_fraction * (
            configured_max_tilt - np.radians(20.0)
        )
        horizontal_limit = acceleration[2] * np.tan(allowed_tilt)
        if horizontal > horizontal_limit:
            acceleration[:2] *= horizontal_limit / horizontal
        return acceleration

    def _guidance(self, observation, position, velocity):
        target, target_velocity = self._select_target(observation, position, velocity)
        if target is None:
            return np.array([0.0, 0.0, 1.0]), 0.8

        now = float(observation["simulation_time"])
        relative = target[:3] - position
        bias_applies = (
            self.guidance_bias_targets is None
            or self.target_index in self.guidance_bias_targets
        )
        if now >= self.guidance_bias_start and bias_applies:
            relative = relative + self.guidance_bias
        distance = np.linalg.norm(relative)
        terminal_distance = (
            self.terminal_distance_late
            if now >= self.terminal_switch_time
            else self.terminal_distance
        )
        if (
            distance <= terminal_distance
            and now >= self.terminal_bias_start
        ):
            relative = relative + self.terminal_bias
            distance = np.linalg.norm(relative)
        line_of_sight = relative / max(distance, 1e-9)
        if self.guidance_mode == "pursuit" and distance > terminal_distance:
            commanded_speed = min(
                self.cruise_speed,
                max(self.approach_speed, 0.9 * distance),
            )
            desired_velocity = target_velocity + commanded_speed * line_of_sight
            acceleration = self.velocity_gain * (desired_velocity - velocity)
            acceleration += np.array([0.0, 0.0, 9.80665])
            acceleration = self._limit_tilt(
                acceleration, float(observation["simulation_time"])
            )
            thrust_acceleration = np.linalg.norm(acceleration)
            desired_axis = acceleration / max(thrust_acceleration, 1e-9)
            self.last_desired_axis = desired_axis.copy()
            throttle = float(np.clip(thrust_acceleration / 12.0, 0.45, 1.0))
            if float(observation["simulation_time"]) - self.min_launch_time < 10.0:
                throttle = 1.0
            return desired_axis, throttle

        relative_velocity = target_velocity - velocity
        closing_speed = max(0.0, -np.dot(relative_velocity, line_of_sight))
        reference_closing_speed = max(
            self.approach_speed,
            min(self.cruise_speed, max(closing_speed, 0.6 * self.cruise_speed)),
        )
        time_to_go = np.clip(distance / reference_closing_speed, 0.35, 6.0)
        zero_effort_miss = relative + relative_velocity * time_to_go
        acceleration = (
            self.velocity_gain * 2.0 * zero_effort_miss / (time_to_go**2)
        )
        acceleration += np.array([0.0, 0.0, 9.80665])
        acceleration = self._limit_tilt(
            acceleration, float(observation["simulation_time"])
        )
        thrust_acceleration = np.linalg.norm(acceleration)
        desired_axis = acceleration / max(thrust_acceleration, 1e-9)
        self.last_desired_axis = desired_axis.copy()
        throttle = float(np.clip(thrust_acceleration / 12.0, 0.45, 1.0))
        if float(observation["simulation_time"]) - self.min_launch_time < 10.0:
            throttle = 1.0
        return desired_axis, throttle

    def _attitude_control(self, desired_axis, gyro, now):
        body_axis = _rotate_body_to_world(
            self.quaternion, np.array([0.0, 0.0, 1.0])
        )
        rotation_error_world = np.cross(body_axis, desired_axis)
        rotation_error_body = _rotate_world_to_body(
            self.quaternion, rotation_error_world
        )
        attitude_gain = (
            self.attitude_gain_late if now >= self.attitude_switch_time else self.attitude_gain
        )
        max_body_rate = (
            self.max_body_rate_late if now >= self.attitude_switch_time else self.max_body_rate
        )
        desired_rates = np.clip(
            attitude_gain * rotation_error_body[:2],
            -max_body_rate,
            max_body_rate,
        )
        tvc = self.rate_gain * (desired_rates - gyro[:2])
        tvc = np.clip(tvc, -self.max_gimbal, self.max_gimbal)
        roll = float(np.clip(-20.0 * gyro[2], -self.max_roll_torque, self.max_roll_torque))
        return tvc, roll

    def get_action(self, observation):
        now = float(observation["simulation_time"])
        states, _, released = self._released_states(observation)
        has_target = released.size > 0 or self.initial_target_index is not None
        wants_launch = not self.launched and now >= self.min_launch_time and has_target
        if wants_launch:
            self.launch_attitude = self._initial_attitude(observation)
            attitude = get_initial_attitude(*self.launch_attitude)
            self.quaternion = np.asarray(attitude, dtype=float)
            self.launch_axis = _rotate_body_to_world(
                self.quaternion, np.array([0.0, 0.0, 1.0])
            )
            self.launched = True

        sensors = np.asarray(observation["rocket_sensors"], dtype=float)
        if self.launched and np.all(np.isfinite(sensors[:3])):
            gyro = sensors[:3]
            self._update_attitude(gyro)
            position, velocity = self._rocket_state(observation)
            if now - self.min_launch_time < self.initial_coast_time:
                tvc = np.clip(
                    -100.0 * gyro[:2], -self.max_gimbal, self.max_gimbal
                )
                roll = float(
                    np.clip(
                        -20.0 * gyro[2],
                        -self.max_roll_torque,
                        self.max_roll_torque,
                    )
                )
                throttle = self.coast_throttle
            else:
                desired_axis, throttle = self._guidance(
                    observation, position, velocity
                )
                tvc, roll = self._attitude_control(desired_axis, gyro, now)
        else:
            tvc, roll, throttle = np.zeros(2), 0.0, 1.0

        return {
            "launch": self.launched,
            "launch_inclination_heading": self.launch_attitude,
            "tvc": tvc,
            "roll": roll,
            "throttle": throttle,
        }
