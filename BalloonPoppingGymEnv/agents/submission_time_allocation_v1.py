"""Observation-only joint-derivative chain guidance.

Generated from controller and online planner source by
scripts/build_chain_submission_agent.py. No training data, stored routes,
seed lookups, or simulator internals are used.
"""
from functools import lru_cache
import numpy as np
from scipy.optimize import minimize
from BalloonPoppingGymEnv.agents.base_agent import BaseAgent


import copy

def get_initial_attitude(inclination, heading):
    """Quaternion from the commanded launch angles (zero initial bank)."""
    theta = np.radians(inclination - 90.0) / 2.0
    psi = np.radians(-heading) / 2.0
    return np.array([np.cos(theta) * np.cos(psi), np.sin(theta) * np.cos(psi), np.sin(theta) * np.sin(psi), np.cos(theta) * np.sin(psi)])

def _multiply_quaternions(left, right):
    """Hamilton product for quaternions stored as [w, x, y, z]."""
    w1, x1, y1, z1 = left
    w2, x2, y2, z2 = right
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2, w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])

def _rotate_body_to_world(quaternion, vector):
    """Rotate a three-vector with q * [0, v] * conjugate(q)."""
    pure = np.concatenate(([0.0], np.asarray(vector, dtype=float)))
    conjugate = quaternion * np.array([1.0, -1.0, -1.0, -1.0])
    return _multiply_quaternions(_multiply_quaternions(quaternion, pure), conjugate)[1:]

def _rotate_world_to_body(quaternion, vector):
    conjugate = quaternion * np.array([1.0, -1.0, -1.0, -1.0])
    pure = np.concatenate(([0.0], np.asarray(vector, dtype=float)))
    return _multiply_quaternions(_multiply_quaternions(conjugate, pure), quaternion)[1:]

G = np.array([0.0, 0.0, -9.80665])

def quintic_intercept(position, velocity, acceleration, target, target_velocity, duration):
    """Minimum integrated squared jerk for fixed endpoint position/velocity/accel.

    Coefficients use normalized time s=t/duration, avoiding small/large-time
    Vandermonde systems. Terminal acceleration is zero (constant balloon drift).
    """
    t = float(duration)
    if not np.isfinite(t) or t <= 0:
        raise ValueError('duration must be finite and positive')
    c0 = np.asarray(position, dtype=float)
    c1 = np.asarray(velocity, dtype=float) * t
    c2 = np.asarray(acceleration, dtype=float) * t * t / 2
    dp = np.asarray(target, dtype=float) - c0 - c1 - c2
    dv = np.asarray(target_velocity, dtype=float) * t - c1 - 2 * c2
    da = -2 * c2
    return np.stack((c0, c1, c2, 10 * dp - 4 * dv + da / 2, -15 * dp + 7 * dv - da, 6 * dp - 3 * dv + da / 2))

def sample_curve(coefficients, duration, elapsed):
    s = np.asarray(elapsed, dtype=float) / duration
    values = []
    c = coefficients.copy()
    for derivative in range(4):
        powers = s[..., None] ** np.arange(len(c))
        values.append(powers @ c / duration ** derivative)
        c = c[1:] * np.arange(1, len(c))[:, None]
    return tuple(values)

def allocate_acceleration(requested, available, max_tilt):
    """Keep requested vertical support, then allocate remaining thrust sideways.

    Unlike normalizing an overlarge vector, this does not silently reduce its
    vertical component. Any saturation still makes the original path infeasible.
    """
    command = np.asarray(requested, dtype=float).copy()
    available = max(float(available), 0.0)
    command[2] = np.clip(command[2], 0.0, available)
    lateral_limit = min(np.sqrt(max(0.0, available ** 2 - command[2] ** 2)), command[2] * np.tan(max_tilt))
    lateral = np.linalg.norm(command[:2])
    if lateral > lateral_limit:
        command[:2] *= lateral_limit / lateral
    return command

def compensated_actuator_command(desired, previous_output, time_constant, rate_limit, timestep, lower, upper):
    """Invert the published first-order lag while tracking the actuator output."""
    desired = float(np.clip(desired, lower, upper))
    previous_output = float(previous_output)
    max_change = float(rate_limit) * float(timestep)
    desired_output = previous_output + np.clip(desired - previous_output, -max_change, max_change)
    alpha = 1.0 if time_constant is None or float(time_constant) == 0 else float(timestep) / (float(time_constant) + float(timestep))
    command = (desired_output - (1.0 - alpha) * previous_output) / alpha
    command = float(np.clip(command, lower, upper))
    filtered = alpha * command + (1.0 - alpha) * previous_output
    output = previous_output + np.clip(filtered - previous_output, -max_change, max_change)
    return (command, float(np.clip(output, lower, upper)))

class PhysicsGuidanceAgent(BaseAgent):
    """Replan smooth intercepts; infer all feedback from permitted sensors."""

    def __init__(self, given_parameters, launch_time=4.0, replan_interval=0.4, max_tilt=55.0, max_axis_rate=0.7, attitude_frequency=None, tracking_frequency=None, horizon=12.0, arrival_speed=0.0, lookahead_velocity=False, minimum_vertical_acceleration=None, direct_rate_gain=None, force_full_throttle=False, position_filter_tau=None, velocity_filter_tau=None, gyro_filter_tau=None):
        super().__init__(given_parameters)
        self.dt = float(given_parameters['simulation']['time_step'])
        self.elevation = float(given_parameters['environment']['elevation'])
        self.radius = float(given_parameters['balloon']['radius'])
        rocket = given_parameters['rocket']
        self.control = rocket['control']
        body, motor, tank = (rocket['rocket_body'], rocket['motor'], rocket['tank'])
        grain_mass = np.pi * (motor['grain_outer_radius'] ** 2 - motor['grain_initial_inner_radius'] ** 2) * motor['grain_initial_height'] * motor['grain_density'] * motor['grain_number']
        self.dry_mass = float(body['mass'] + motor['dry_mass'])
        self.initial_mass = self.dry_mass + grain_mass + tank['initial_liquid_mass'] + tank['initial_gas_mass']
        self.burn_time = float(motor['burn_time'])
        self.mass_flow = tank['liquid_mass_flow_rate_out'] + grain_mass / self.burn_time
        self.thrust = float(motor['thrust_source'])
        tank_z = tank['tank_position'] + motor['motor_position']
        tank_mass = tank['initial_liquid_mass'] + tank['initial_gas_mass']
        self.inertia = np.asarray(body['inertia'], dtype=float).copy()
        self.inertia[:2] += tank_mass * tank_z ** 2
        self.lever = max(abs(motor['motor_position'] + motor['nozzle_position']), 0.1)
        self.launch_time = float(launch_time)
        self.replan_interval = float(replan_interval)
        self.max_tilt = np.radians(max_tilt)
        self.max_axis_rate = float(max_axis_rate)
        actuator_lag = self.control['gimbal_time_constant'] not in (None, 0)
        self.attitude_frequency = float(3.0 if attitude_frequency is None and actuator_lag else 4.0 if attitude_frequency is None else attitude_frequency)
        self.tracking_frequency = float(1.3 if tracking_frequency is None and actuator_lag else 1.0 if tracking_frequency is None else tracking_frequency)
        self.horizon = float(horizon)
        self.arrival_speed = float(arrival_speed)
        self.lookahead_velocity = bool(lookahead_velocity)
        self.minimum_vertical_acceleration = None if minimum_vertical_acceleration is None else float(minimum_vertical_acceleration)
        self.direct_rate_gain = None if direct_rate_gain is None else float(direct_rate_gain)
        self.force_full_throttle = bool(force_full_throttle)
        sensors = rocket['sensors']
        noisy_position = max(float(sensors['gnss_position_accuracy']), float(sensors['gnss_altitude_accuracy'])) > 0
        noisy_velocity = float(sensors['gnss_velocity_accuracy']) > 0
        noisy_gyro = float(sensors['gyro_noise_density']) > 0 or float(sensors['gyro_random_walk_density']) > 0
        self.position_filter_tau = float(0.5 if position_filter_tau is None and noisy_position else 0.0 if position_filter_tau is None else position_filter_tau)
        self.velocity_filter_tau = float(0.04 if velocity_filter_tau is None and noisy_velocity else 0.0 if velocity_filter_tau is None else velocity_filter_tau)
        self.gyro_filter_tau = float(0.02 if gyro_filter_tau is None and noisy_gyro else 0.0 if gyro_filter_tau is None else gyro_filter_tau)
        if min(self.position_filter_tau, self.velocity_filter_tau, self.gyro_filter_tau) < 0:
            raise ValueError('Sensor filter time constants cannot be negative')
        self.launched = False
        self.launch_attitude = np.array([90.0, 0.0])
        self.quaternion = np.asarray(get_initial_attitude(90.0, 0.0))
        self.previous_gyro = np.zeros(3)
        self.filtered_gyro = None
        self.filtered_position = None
        self.filtered_velocity = None
        self.previous_velocity = None
        self.acceleration = np.zeros(3)
        self.disturbance = np.zeros(3)
        self.previous_tvc = np.zeros(2)
        self.previous_roll = 0.0
        self.previous_throttle = 1.0
        self.target_index = None
        self.target_events = []
        self.plan = None
        self.plan_start = 0.0
        self.plan_duration = 0.0
        self.next_plan = 0.0
        self.last_desired_axis = np.array([0.0, 0.0, 1.0])
        self.failed_until = {}
        self.diagnostics = {'plans': 0, 'no_feasible_plan': 0, 'saturated_steps': 0, 'position_error_sum': 0.0, 'tracking_steps': 0, 'position_innovation_sum': 0.0, 'navigation_updates': 0}

    def _filter_sensors(self, gyro, position, velocity):
        """Filter only observations, using known launch position for initialization."""
        gyro = np.asarray(gyro, dtype=float)
        position = np.asarray(position, dtype=float)
        velocity = np.asarray(velocity, dtype=float)
        gyro_gain = 1.0 if self.gyro_filter_tau == 0 else -np.expm1(-self.dt / self.gyro_filter_tau)
        position_gain = 1.0 if self.position_filter_tau == 0 else -np.expm1(-self.dt / self.position_filter_tau)
        velocity_gain = 1.0 if self.velocity_filter_tau == 0 else -np.expm1(-self.dt / self.velocity_filter_tau)
        if self.filtered_gyro is None:
            self.filtered_gyro = gyro.copy() if gyro_gain == 1 else np.zeros(3)
        else:
            self.filtered_gyro += gyro_gain * (gyro - self.filtered_gyro)
        if self.filtered_position is None:
            self.filtered_position = position.copy() if position_gain == 1 else np.array([0.0, 0.0, self.elevation])
            self.filtered_velocity = velocity.copy()
        else:
            predicted_position = self.filtered_position + self.filtered_velocity * self.dt
            innovation = position - predicted_position
            self.diagnostics['position_innovation_sum'] += float(np.linalg.norm(innovation))
            self.diagnostics['navigation_updates'] += 1
            self.filtered_position = predicted_position + position_gain * innovation
            self.filtered_velocity += velocity_gain * (velocity - self.filtered_velocity)
        return (self.filtered_gyro.copy(), self.filtered_position.copy(), self.filtered_velocity.copy())

    def available_acceleration(self, now):
        elapsed = max(0.0, now - self.launch_time)
        if elapsed >= self.burn_time:
            return 0.0
        mass = max(self.dry_mass, self.initial_mass - self.mass_flow * elapsed)
        return self.thrust / mass

    def _feasible(self, c, duration, now):
        times = np.linspace(0.0, duration, 33)
        p, v, a, jerk = sample_curve(c, duration, times)
        thrust_accel = a - G - self.disturbance
        magnitude = np.linalg.norm(thrust_accel, axis=1)
        limits = np.array([self.available_acceleration(now + t) for t in times])
        if np.any(magnitude > limits) or np.any(magnitude < 0.5):
            return False
        if np.any(thrust_accel[:, 2] < magnitude * np.cos(self.max_tilt)):
            return False
        axes = thrust_accel / magnitude[:, None]
        axis_rate = np.linalg.norm(jerk - axes * np.sum(axes * jerk, axis=1)[:, None], axis=1) / magnitude
        if np.any(axis_rate > self.max_axis_rate):
            return False
        throttle = magnitude / np.maximum(limits, 1e-09)
        if np.any(np.abs(np.diff(throttle) / np.diff(times)) > self.control['throttle_rate_limit']):
            return False
        return bool(np.all(p[:, 2] >= self.elevation - 0.05))

    def _make_plan(self, observation, position, velocity, now):
        states = np.asarray(observation['balloon_states'], dtype=float)
        status = np.asarray(observation['balloon_status']).reshape(-1)
        released = np.flatnonzero(status == 1)
        moving = released[np.linalg.norm(states[released, 3:6], axis=1) > 0.5]
        shared_velocity = np.median(states[moving, 3:6], axis=0) if moving.size else np.zeros(3)
        candidates = [int(i) for i in released if self.failed_until.get(int(i), 0) <= now]
        locked = self.target_index in candidates
        if locked:
            candidates = [self.target_index]
        else:
            candidates.sort(key=lambda i: np.linalg.norm(states[i, :3] - position))
            candidates = candidates[:12]
        earliest = max(0.4, self.plan_start + self.plan_duration - now) if locked else 0.8
        best = None
        for index in candidates:
            drift = states[index, 3:6]
            if np.linalg.norm(drift) <= 0.5:
                drift = shared_velocity
            relative = states[index, :3] - position
            direction = relative / max(np.linalg.norm(relative), 1e-09)
            terminal_velocity = drift + self.arrival_speed * direction
            durations = np.unique(np.r_[earliest, np.arange(max(earliest, 0.8), self.horizon + 0.01, 0.4)])
            for duration in durations:
                if best is not None and duration >= best[0]:
                    break
                target = states[index, :3] + drift * duration
                if self.lookahead_velocity and self.arrival_speed > 0:
                    others = released[released != index]
                    if others.size:
                        next_positions = states[others, :3] + states[others, 3:6] * duration
                        deltas = next_positions - target
                        distances = np.linalg.norm(deltas, axis=1)
                        nearest = int(np.argmin(distances))
                        reserve = np.sqrt(max(0.0, self.available_acceleration(now + duration) ** 2 - G[2] ** 2))
                        speed = min(self.arrival_speed, np.sqrt(reserve * distances[nearest]))
                        terminal_velocity = drift + speed * deltas[nearest] / max(distances[nearest], 1e-09)
                c = quintic_intercept(position, velocity, self.acceleration, target, terminal_velocity, duration)
                if self._feasible(c, duration, now):
                    best = (duration, index, c)
                    break
        if best is None:
            self.diagnostics['no_feasible_plan'] += 1
            if self.plan is not None and now < self.plan_start + self.plan_duration:
                return
            if self.target_index is not None:
                self.failed_until[self.target_index] = now + 2 * self.replan_interval
            self.target_index = None
            self.plan = None
            return
        duration, index, c = best
        if index != self.target_index:
            self.target_events.append((now, index))
        self.target_index = index
        self.plan, self.plan_start, self.plan_duration = (c, now, duration)
        self.diagnostics['plans'] += 1

    def _attitude_action(self, axis, jerk, magnitude, gyro, available):
        body_axis = _rotate_body_to_world(self.quaternion, [0.0, 0.0, 1.0])
        error = _rotate_world_to_body(self.quaternion, np.cross(body_axis, axis))
        omega_world = np.cross(axis, jerk / max(magnitude, 1.0))
        omega_ref = _rotate_world_to_body(self.quaternion, omega_world)
        omega_ref = np.clip(omega_ref, -self.max_axis_rate, self.max_axis_rate)
        wn = self.attitude_frequency
        if self.direct_rate_gain is None:
            angular_accel = wn ** 2 * error + 2 * wn * (omega_ref - gyro)
            effective_force = self.thrust * max(self.previous_throttle, 0.05)
            torque = self.inertia * angular_accel
            tvc = np.degrees(np.arcsin(np.clip(torque[:2] / (effective_force * self.lever), -1.0, 1.0)))
        else:
            desired_rate = np.clip(omega_ref[:2] + wn * error[:2], -self.max_axis_rate, self.max_axis_rate)
            tvc = self.direct_rate_gain * (desired_rate - gyro[:2])
        tvc = np.clip(tvc, -self.control['max_gimbal_angle'], self.control['max_gimbal_angle'])
        tvc_commands = np.zeros(2)
        tvc_outputs = np.zeros(2)
        for index in range(2):
            tvc_commands[index], tvc_outputs[index] = compensated_actuator_command(tvc[index], self.previous_tvc[index], self.control['gimbal_time_constant'], self.control['gimbal_rate_limit'], self.dt, -self.control['max_gimbal_angle'], self.control['max_gimbal_angle'])
        desired_throttle = 1.0 if self.force_full_throttle else magnitude / max(available, 1e-09)
        throttle, throttle_output = compensated_actuator_command(desired_throttle, self.previous_throttle, self.control['throttle_time_constant'], self.control['throttle_rate_limit'], self.dt, *self.control['throttle_range'])
        desired_roll = float(np.clip(-2 * wn * self.inertia[2] * gyro[2], -self.control['max_roll_torque'], self.control['max_roll_torque']))
        roll, roll_output = compensated_actuator_command(desired_roll, self.previous_roll, self.control['roll_torque_time_constant'], self.control['torque_rate_limit'], self.dt, -self.control['max_roll_torque'], self.control['max_roll_torque'])
        self.previous_tvc = tvc_outputs
        self.previous_roll = roll_output
        self.previous_throttle = throttle_output
        return (tvc_commands, roll, throttle)

    def get_action(self, observation):
        now = float(observation['simulation_time'])
        if not self.launched and now >= self.launch_time:
            self.launched = True
            self.launch_time = now + self.dt
        sensors = np.asarray(observation['rocket_sensors'], dtype=float)
        tvc, roll, throttle = (np.zeros(2), 0.0, 1.0)
        if self.launched and np.all(np.isfinite(sensors)):
            gyro, position, velocity = self._filter_sensors(sensors[:3], sensors[6:9], sensors[9:12])
            increment = (gyro + self.previous_gyro) * self.dt / 2
            angle = np.linalg.norm(increment)
            if angle > 1e-12:
                dq = np.r_[np.cos(angle / 2), np.sin(angle / 2) * increment / angle]
                self.quaternion = _multiply_quaternions(self.quaternion, dq)
                self.quaternion /= np.linalg.norm(self.quaternion)
            self.previous_gyro = gyro.copy()
            available = self.available_acceleration(now)
            if self.previous_velocity is not None:
                measured = (velocity - self.previous_velocity) / self.dt
                self.acceleration += self.dt / (0.08 + self.dt) * (measured - self.acceleration)
                body_axis = _rotate_body_to_world(self.quaternion, [0, 0, 1])
                residual = measured - G - available * self.previous_throttle * body_axis
                self.disturbance += self.dt / (0.8 + self.dt) * (np.clip(residual, -3, 3) - self.disturbance)
            else:
                body_axis = _rotate_body_to_world(self.quaternion, [0, 0, 1])
                self.acceleration = G + available * body_axis
            self.previous_velocity = velocity.copy()
            status = np.asarray(observation['balloon_status']).reshape(-1)
            if self.target_index is not None and status[self.target_index] != 1:
                self.target_index = None
                self.plan = None
                self.next_plan = now
            if now >= self.next_plan:
                self._make_plan(observation, position, velocity, now)
                self.next_plan = now + self.replan_interval
            if self.plan is not None:
                elapsed = np.clip(now - self.plan_start, 0, self.plan_duration)
                p, v, a, jerk = sample_curve(self.plan, self.plan_duration, elapsed)
                wn = self.tracking_frequency
                acceleration = a + wn ** 2 * (p - position) + 2 * wn * (v - velocity)
                self.diagnostics['position_error_sum'] += float(np.linalg.norm(p - position))
                self.diagnostics['tracking_steps'] += 1
            else:
                acceleration = np.array([0.0, 0.0, max(0.0, -velocity[2])])
                jerk = np.zeros(3)
            requested = acceleration - G - self.disturbance
            if self.minimum_vertical_acceleration is not None and available > 0:
                requested[2] = max(requested[2], -G[2] + self.minimum_vertical_acceleration)
            allocated = allocate_acceleration(requested, available, self.max_tilt)
            if np.linalg.norm(allocated - requested) > 0.1:
                self.diagnostics['saturated_steps'] += 1
            magnitude = np.linalg.norm(allocated)
            axis = allocated / max(magnitude, 1e-09) if magnitude > 1e-09 else np.array([0.0, 0.0, 1.0])
            self.last_desired_axis = axis.copy()
            tvc, roll, throttle = self._attitude_action(axis, jerk, magnitude, gyro, available)
        return {'launch': self.launched, 'launch_inclination_heading': self.launch_attitude.copy(), 'tvc': tvc, 'roll': roll, 'throttle': throttle}

def boundary_curve(p, v, a, end_p, end_v, end_a, duration):
    c = quintic_intercept(p, v, a, end_p, end_v, duration)
    correction = duration ** 2 * np.asarray(end_a)
    c[3] += correction / 2
    c[4] -= correction
    c[5] += correction / 2
    return c

@lru_cache(maxsize=512)
def spline_matrices(first_time, second_time):
    z, one = (np.zeros(1), np.ones(1))
    d1 = np.column_stack((boundary_curve(z, z, z, z, one, z, first_time)[:, 0], boundary_curve(z, z, z, z, z, one, first_time)[:, 0]))
    d2 = np.column_stack((boundary_curve(z, one, z, z, z, z, second_time)[:, 0], boundary_curve(z, z, one, z, z, z, second_time)[:, 0]))
    factor = np.array([6.0, 24.0, 60.0])
    gram = np.outer(factor, factor) / (np.arange(3)[:, None] + np.arange(3)[None, :] + 1)
    q1, q2 = (gram / first_time ** 5, gram / second_time ** 5)
    left, right = (d1[3:].T @ q1, d2[3:].T @ q2)
    hessian = left @ d1[3:] + right @ d2[3:]
    return (d1, d2, left, right, np.linalg.inv(hessian))

def two_intercept_spline(p, v, a, first_p, second_p, final_v, first_time, second_time):
    """Exact minimum squared jerk over two quintics with a C2 interior knot."""
    if first_time <= 0 or second_time <= 0:
        raise ValueError('Segment durations must be positive')
    zero = np.zeros(3)
    b1 = boundary_curve(p, v, a, first_p, zero, zero, first_time)
    b2 = boundary_curve(first_p, zero, zero, second_p, final_v, zero, second_time)
    d1, d2, left, right, inverse = spline_matrices(float(first_time), float(second_time))
    interior = -inverse @ (left @ b1[3:] + right @ b2[3:])
    return (b1 + d1 @ interior, b2 + d2 @ interior)

class SplineRouteAgent(PhysicsGuidanceAgent):

    def __init__(self, given_parameters, first_candidates=8, next_candidates=3, second_horizon=8.0, duration_step=1.0, **kwargs):
        super().__init__(given_parameters, **kwargs)
        self.first_candidates = int(first_candidates)
        self.next_candidates = int(next_candidates)
        self.second_horizon = float(second_horizon)
        self.duration_step = float(duration_step)
        if min(self.first_candidates, self.next_candidates, self.second_horizon, self.duration_step) <= 0:
            raise ValueError('Search limits must be positive')
        self.diagnostics.update(pair_candidates=0, feasible_pairs=0, pair_plans=0, fallback_plans=0)
        self.route_events = []

    def _make_plan(self, observation, position, velocity, now):
        states = np.asarray(observation['balloon_states'], dtype=float)
        released = np.flatnonzero(np.asarray(observation['balloon_status']).reshape(-1) == 1)
        moving = released[np.linalg.norm(states[released, 3:6], axis=1) > 0.5]
        shared = np.median(states[moving, 3:6], axis=0) if moving.size else np.zeros(3)
        drift = states[:, 3:6].copy()
        drift[np.linalg.norm(drift, axis=1) <= 0.5] = shared
        candidates = [int(i) for i in released if self.failed_until.get(int(i), 0) <= now]
        locked = self.target_index in candidates
        if locked:
            candidates = [self.target_index]
            eta = max(0.4, self.plan_start + self.plan_duration - now)
            first_times = np.array([eta, eta + 0.8, eta + 1.6])
        else:
            candidates.sort(key=lambda i: np.linalg.norm(states[i, :3] - position))
            candidates = candidates[:self.first_candidates]
            first_times = np.arange(2.0, self.horizon + 0.01, self.duration_step)
        remaining = self.burn_time - (now - self.launch_time) - 1e-05
        best = None
        for first_time in first_times:
            if first_time >= remaining:
                continue
            if best is not None and first_time + 1.5 >= best[0]:
                continue
            predicted = states[:, :3] + drift * first_time
            for first in candidates:
                others = [int(j) for j in released if j != first]
                others.sort(key=lambda j: np.linalg.norm(predicted[j] - predicted[first]))
                for second in others[:self.next_candidates]:
                    for second_time in np.arange(1.5, self.second_horizon + 0.01, self.duration_step):
                        total = first_time + second_time
                        if total > remaining or (best is not None and total >= best[0]):
                            break
                        self.diagnostics['pair_candidates'] += 1
                        c1, c2 = two_intercept_spline(position, velocity, self.acceleration, predicted[first], states[second, :3] + drift[second] * total, drift[second], first_time, second_time)
                        if not self._feasible(c1, first_time, now):
                            continue
                        if not self._feasible(c2, second_time, now + first_time):
                            continue
                        self.diagnostics['feasible_pairs'] += 1
                        best = (total, first, second, first_time, c1)
                        break
        if best is None:
            super()._make_plan(observation, position, velocity, now)
            self.diagnostics['fallback_plans'] += 1
            return
        total, first, second, duration, curve = best
        if first != self.target_index:
            self.target_events.append((now, first))
        self.route_events.append([float(now), int(first), int(second), float(duration), float(total)])
        self.target_index = first
        self.plan, self.plan_start, self.plan_duration = (curve, now, float(duration))
        self.diagnostics['plans'] += 1
        self.diagnostics['pair_plans'] += 1

def batch_samples(curves, durations, samples=25):
    s = np.linspace(0.0, 1.0, samples)
    c = curves
    result = []
    for order in range(4):
        basis = s[:, None] ** np.arange(c.shape[1])
        result.append(np.einsum('sk,nkj->nsj', basis, c) / np.asarray(durations)[:, None, None] ** order)
        c = c[:, 1:] * np.arange(1, c.shape[1])[None, :, None]
    return result

class RouteOptimizerMixin:

    def _curves(self, position, velocity, states, drift, route, durations):
        arrivals = np.cumsum(durations)
        targets = states[route, :3] + drift[route] * arrivals[:, None]
        curves = []
        p, v, a = (position, velocity, self.acceleration)
        for target, end_v, end_a, duration in zip(targets, self.endpoint_velocities, self.endpoint_accelerations, durations):
            curves.append(boundary_curve(p, v, a, target, end_v, end_a, duration))
            p, v, a = (target, end_v, end_a)
        return np.asarray(curves)

    def _margins(self, curves, durations, now, samples=33):
        durations = np.asarray(durations, dtype=float)
        p, _, a, j = batch_samples(curves, durations, samples)
        starts = np.r_[0.0, np.cumsum(durations)[:-1]]
        times = now + starts[:, None] + durations[:, None] * np.linspace(0.0, 1.0, samples)
        age = np.maximum(times - self.launch_time, 0.0)
        limit = self.thrust / np.maximum(self.dry_mass, self.initial_mass - self.mass_flow * age)
        force = a - G - self.disturbance
        mag = np.linalg.norm(force, axis=2)
        axis = force / np.maximum(mag[:, :, None], 1e-09)
        rates = np.linalg.norm(j - axis * np.sum(axis * j, axis=2)[:, :, None], axis=2) / np.maximum(mag, 1e-09)
        throttle = mag / limit
        throttle_change = np.diff(throttle, axis=1) / np.diff(times, axis=1)
        return np.r_[(limit - mag).ravel() / 12.0, (mag - 0.5).ravel() / 12.0, (axis[:, :, 2] - np.cos(self.max_tilt)).ravel(), (self.max_axis_rate ** 2 - rates ** 2).ravel(), (p[:, :, 2] - self.elevation + 0.05).ravel() / 10.0, (self.control['throttle_rate_limit'] ** 2 - throttle_change ** 2).ravel(), self.launch_time + self.burn_time - now - sum(durations) - 0.0001]

    def _solve(self, position, velocity, states, drift, route, initial, now, held):
        self.diagnostics['joint_calls'] += 1
        count = len(initial)
        free = self.optimize_derivatives and (not held)
        seed_v, seed_a = (self.endpoint_velocities.copy(), self.endpoint_accelerations.copy())
        seed = np.r_[initial, seed_v.ravel() / 10.0, seed_a.ravel() / 5.0] if free else initial

        def unpack(x):
            if free:
                self.endpoint_velocities = x[count:count + 3 * count].reshape(count, 3) * 10.0
                self.endpoint_accelerations = x[count + 3 * count:].reshape(count, 3) * 5.0
            return x[:count]

        def curves(x):
            durations = unpack(x)
            return self._curves(position, velocity, states, drift, route, durations)

        def margins(x):
            return self._margins(curves(x), x[:count], now)

        def objective(x):
            return sum(x[:count]) + (4.0 * max(0.0, x[0] - initial[0]) if held else 0.0)
        bounds = [(max(0.15, t - 2.0), t + 2.0) for t in initial]
        if free:
            bounds += [(-6.0, 6.0)] * (3 * count) + [(-2.0, 2.0)] * (3 * count)
        result = minimize(objective, seed, method='SLSQP', bounds=bounds, constraints={'type': 'ineq', 'fun': margins}, options={'maxiter': self.solver_iterations, 'ftol': 1e-08})
        choices = [seed]
        if np.all(np.isfinite(result.x)):
            choices.extend((seed + fraction * (result.x - seed) for fraction in (1.0, 0.99, 0.97, 0.95, 0.9, 0.8, 0.6, 0.4)))
        choices.sort(key=objective)
        for x in choices:
            if np.min(margins(x)) < -1e-06:
                continue
            cs = curves(x)
            durations = x[:count]
            if np.min(self._margins(cs, durations, now, samples=65)) >= -1e-06:
                self.diagnostics['joint_accepted'] += 1
                return (cs, np.asarray(durations))
        self.endpoint_velocities, self.endpoint_accelerations = (seed_v, seed_a)
        return None

class OptimizedSplineAgent(SplineRouteAgent):
    _curves = RouteOptimizerMixin._curves
    _margins = RouteOptimizerMixin._margins
    _solve = RouteOptimizerMixin._solve

    def __init__(self, given_parameters, solver_iterations=40, **kwargs):
        super().__init__(given_parameters, **kwargs)
        self.solver_iterations = int(solver_iterations)
        self.optimize_derivatives = True
        self.endpoint_velocities = self.endpoint_accelerations = None
        self.diagnostics.update(joint_calls=0, joint_accepted=0)

    def _make_plan(self, observation, position, velocity, now):
        old_target = self.target_index
        previous = len(self.route_events)
        super()._make_plan(observation, position, velocity, now)
        if len(self.route_events) == previous:
            return
        _, first, second, t1, total = self.route_events[-1]
        states = np.asarray(observation['balloon_states'], dtype=float)
        released = np.flatnonzero(np.asarray(observation['balloon_status']).reshape(-1) == 1)
        drift = states[:, 3:6].copy()
        moving = released[np.linalg.norm(drift[released], axis=1) > 0.5]
        if moving.size:
            drift[np.linalg.norm(drift, axis=1) <= 0.5] = np.median(drift[moving], axis=0)
        times = np.array([t1, total - t1])
        curves = np.asarray(two_intercept_spline(position, velocity, self.acceleration, states[first, :3] + drift[first] * t1, states[second, :3] + drift[second] * total, drift[second], *times))
        _, vs, acs, _ = batch_samples(curves, times)
        self.endpoint_velocities, self.endpoint_accelerations = (vs[:, -1], acs[:, -1])
        solved = self._solve(position, velocity, states, drift, [first, second], times, now, old_target == first)
        if solved is not None:
            curves, times = solved
            self.plan, self.plan_start, self.plan_duration = (curves[0], now, float(times[0]))

class SubmissionAgent(OptimizedSplineAgent):
    """Public entry point for the official evaluator."""

def arrival_curves(p, v, a, targets, drift, durations, braking):
    """Minimum jerk with free final derivatives, blended with drift arrivals."""
    t = np.asarray(durations)[:, None]
    c = np.zeros((len(t), 6, 3))
    c[:, 0], c[:, 1], c[:, 2] = (p, v * t, a * t * t / 2)
    dp = targets - c[:, 0] - c[:, 1] - c[:, 2]
    dv = drift * t - c[:, 1] - 2 * c[:, 2]
    da = -2 * c[:, 2]
    natural = np.stack((5 * dp / 3, -5 * dp / 6, dp / 6), axis=1)
    stopped = np.stack((10 * dp - 4 * dv + da / 2, -15 * dp + 7 * dv - da, 6 * dp - 3 * dv + da / 2), axis=1)
    mix = np.asarray(braking)[:, None, None]
    c[:, 3:] = natural * (1 - mix) + stopped * mix
    return c

class MomentumBeamAgent(PhysicsGuidanceAgent):

    def __init__(self, given_parameters, beam_width=36, search_depth=8, branch_targets=12, time_grid=0.5, leg_horizon=10.0, braking_options=(0.0, 0.25, 0.5, 0.75, 1.0), capture_fraction=0.25, reserve=0.0, commitment=True, velocity_bin=5.0, arrival_model='natural', first_quota=0, **kwargs):
        super().__init__(given_parameters, **kwargs)
        self.beam_width = int(beam_width)
        self.search_depth = int(search_depth)
        self.branch_targets = int(branch_targets)
        self.time_grid = float(time_grid)
        self.leg_horizon = float(leg_horizon)
        self.braking_options = np.asarray(braking_options)
        self.capture_fraction = float(capture_fraction)
        self.reserve = float(reserve)
        self.commitment = bool(commitment)
        self.velocity_bin = float(velocity_bin)
        self.arrival_model = str(arrival_model)
        self.first_quota = int(first_quota)
        if min(self.beam_width, self.search_depth, self.branch_targets, self.time_grid, self.leg_horizon, self.velocity_bin) <= 0:
            raise ValueError('Positive search limits required')
        if not 0 <= self.capture_fraction < 1:
            raise ValueError('Capture fraction must be in [0,1)')
        self.route = []
        self.deadlines = []
        self.ends_v = []
        self.ends_a = []
        self.route_events = []
        self.diagnostics.update(beam_searches=0, beam_candidates=0, beam_feasible=0, maximum_depth=0)

    def _valid(self, curves, durations, start, samples=17):
        p, v, a, j = batch_samples(curves, durations, samples)
        force = a - G - self.disturbance
        mag = np.linalg.norm(force, axis=2)
        times = start + durations[:, None] * np.linspace(0, 1, samples)
        age = np.maximum(times - self.launch_time, 0.0)
        limit = self.thrust / np.maximum(self.dry_mass, self.initial_mass - self.mass_flow * age)
        axes = force / np.maximum(mag[:, :, None], 1e-09)
        rate = np.linalg.norm(j - axes * np.sum(axes * j, axis=2)[:, :, None], axis=2) / np.maximum(mag, 1e-09)
        throttle = mag / limit
        reserve = self.reserve * np.linspace(0, 1, samples)
        valid = np.all((mag <= limit - reserve + 1e-06) & (mag >= 0.5) & (axes[:, :, 2] >= np.cos(self.max_tilt)) & (rate <= self.max_axis_rate) & (p[:, :, 2] >= self.elevation - 0.05), axis=1)
        valid &= np.all(np.abs(np.diff(throttle, axis=1)) / np.diff(times, axis=1) <= self.control['throttle_rate_limit'], axis=1)
        return (valid, p[:, -1], v[:, -1], a[:, -1])

    def _make_plan(self, observation, position, velocity, now):
        states = np.asarray(observation['balloon_states'], dtype=float)
        status = np.asarray(observation['balloon_status']).reshape(-1)
        released = np.flatnonzero(status == 1)
        drift = states[:, 3:6]
        while self.route and status[self.route[0]] != 1:
            self.route.pop(0)
            self.deadlines.pop(0)
            self.ends_v.pop(0)
            self.ends_a.pop(0)
        if self.route and (self.commitment or self.target_index == self.route[0]):
            duration = self.deadlines[0] - now
            if duration > 0.12:
                target = states[self.route[0], :3] + drift[self.route[0]] * duration
                c = boundary_curve(position, velocity, self.acceleration, target, self.ends_v[0], self.ends_a[0], duration)
                if self._feasible(c, duration, now):
                    self._select(self.route[0], c, duration, now)
                    return
                if self.plan is not None and self.target_index == self.route[0] and (now < self.plan_start + self.plan_duration):
                    return
            else:
                self.failed_until[self.route[0]] = now + 1.0
        self.route, self.deadlines, self.ends_v, self.ends_a = ([], [], [], [])
        cutoff = self.launch_time + self.burn_time - 0.0001
        beam = [(now, position, velocity, self.acceleration, (), (), (), (), ())]
        best = None
        self.diagnostics['beam_searches'] += 1
        for depth in range(self.search_depth):
            children = []
            for start, p, v, a, route, cs, ts, evs, eas in beam:
                ids = [int(i) for i in released if i not in route and self.failed_until.get(int(i), 0) <= now]
                if not ids:
                    continue
                projected = states[:, :3] + drift * (start - now)
                estimates = []
                for look in (1.0, 2.0, 4.0, 7.0):
                    delta = projected[ids] + drift[ids] * look - p - v * look
                    estimates.append(np.linalg.norm(delta, axis=1) / look ** 2 + look * 0.12)
                order = np.argsort(np.min(estimates, axis=0))[:self.branch_targets]
                ids = np.asarray(ids)[order]
                durations = np.arange(0.5, min(self.leg_horizon, cutoff - start) + 0.001, self.time_grid)
                if not len(durations):
                    continue
                ii, tt, bb = np.meshgrid(ids, durations, self.braking_options, indexing='ij')
                ii, tt, bb = (ii.ravel(), tt.ravel(), bb.ravel())
                targets = projected[ii] + drift[ii] * tt[:, None]
                ballistic = p + v * tt[:, None] + a * tt[:, None] ** 2 / 2
                delta = ballistic - targets
                lengths = np.linalg.norm(delta, axis=1)
                targets += delta * (np.minimum(self.radius * self.capture_fraction, lengths) / np.maximum(lengths, 1e-09))[:, None]
                curves = arrival_curves(p, v, a, targets, drift[ii], tt, bb)
                if self.arrival_model != 'natural':
                    end_v = 2 * (targets - p) / tt[:, None] - v
                    end_v = end_v * (1 - bb[:, None]) + drift[ii] * bb[:, None]
                    end_a = np.zeros_like(end_v)
                    if self.arrival_model == 'climb':
                        end_a[:, 2] = 1.5
                    c0, c1, c2 = (p, v * tt[:, None], a * tt[:, None] ** 2 / 2)
                    dp = targets - c0 - c1 - c2
                    dv = end_v * tt[:, None] - c1 - 2 * c2
                    da = end_a * tt[:, None] ** 2 - 2 * c2
                    curves[:, 3] = 10 * dp - 4 * dv + da / 2
                    curves[:, 4] = -15 * dp + 7 * dv - da
                    curves[:, 5] = 6 * dp - 3 * dv + da / 2
                valid, ends, ev, ea = self._valid(curves, tt, start)
                self.diagnostics['beam_candidates'] += len(tt)
                self.diagnostics['beam_feasible'] += int(valid.sum())
                for k in np.flatnonzero(valid):
                    children.append((start + tt[k], ends[k], ev[k], ea[k], route + (int(ii[k]),), cs + (curves[k],), ts + (float(tt[k]),), evs + (ev[k],), eas + (ea[k],)))
            if not children:
                break
            children.sort(key=lambda n: n[0])
            beam, bins, quotas = ([], set(), {})
            for node in children:
                key = (node[4][0], node[4][-1], tuple(np.round(node[2] / self.velocity_bin).astype(int)))
                if key in bins:
                    continue
                if self.first_quota and quotas.get(node[4][0], 0) >= self.first_quota:
                    continue
                bins.add(key)
                quotas[node[4][0]] = quotas.get(node[4][0], 0) + 1
                beam.append(node)
                if len(beam) >= self.beam_width:
                    break
            best = beam[0]
            self.diagnostics['maximum_depth'] = max(self.diagnostics['maximum_depth'], depth + 1)
        if best is None:
            return super()._make_plan(observation, position, velocity, now)
        finish, _, _, _, route, curves, durations, evs, eas = best
        if not self._feasible(curves[0], durations[0], now):
            return super()._make_plan(observation, position, velocity, now)
        self.route = list(route)
        self.deadlines = list(now + np.cumsum(durations))
        self.ends_v, self.ends_a = (list(evs), list(eas))
        self.route_events.append([now, list(route), float(finish - now)])
        self._select(route[0], curves[0], durations[0], now)

    def _select(self, index, curve, duration, now):
        if self.target_index != index:
            self.target_events.append((now, int(index)))
        self.target_index = int(index)
        self.plan, self.plan_start, self.plan_duration = (curve, now, float(duration))
        self.diagnostics['plans'] += 1

@lru_cache(maxsize=2048)
def free_chain_matrices(durations, terminal_weight):
    count = len(durations)
    size = 2 * count
    zero, one = (np.zeros(1), np.ones(1))
    factors = np.array([6.0, 24.0, 60.0])
    gram = np.outer(factors, factors) / (np.arange(3)[:, None] + np.arange(3)[None, :] + 1)
    designs, projections = ([], [])
    hessian = np.zeros((size, size))
    for i, t in enumerate(durations):
        d = np.zeros((6, size))
        if i:
            d[:, 2 * (i - 1)] = boundary_curve(zero, one, zero, zero, zero, zero, t)[:, 0]
            d[:, 2 * (i - 1) + 1] = boundary_curve(zero, zero, one, zero, zero, zero, t)[:, 0]
        d[:, 2 * i] = boundary_curve(zero, zero, zero, zero, one, zero, t)[:, 0]
        d[:, 2 * i + 1] = boundary_curve(zero, zero, zero, zero, zero, one, t)[:, 0]
        projection = d[3:].T @ (gram / t ** 5)
        hessian += projection @ d[3:]
        designs.append(d)
        projections.append(projection)
    hessian[-2, -2] += terminal_weight
    hessian[-1, -1] += terminal_weight * 4
    return (np.asarray(designs), np.asarray(projections), np.linalg.inv(hessian))

def free_chain(p, v, a, targets, durations, drift, terminal_weight=0.0):
    durations = tuple((float(t) for t in durations))
    zero = np.zeros(3)
    origins = [p] + list(targets[:-1])
    bases = np.array([boundary_curve(origins[i], v if i == 0 else zero, a if i == 0 else zero, targets[i], zero, zero, t) for i, t in enumerate(durations)])
    designs, projections, inverse = free_chain_matrices(durations, float(terminal_weight))
    rhs = np.einsum('nik,nkj->ij', projections, bases[:, 3:])
    rhs[-2] -= terminal_weight * drift
    derivatives = -inverse @ rhs
    return bases + np.einsum('nki,ij->nkj', designs, derivatives)

class ChainBeamAgent(MomentumBeamAgent):

    def __init__(self, given_parameters, terminal_weight=0.1, short_leg_times=(), **kwargs):
        super().__init__(given_parameters, **kwargs)
        self.terminal_weight = float(terminal_weight)
        self.short_leg_times = np.asarray(sorted(set((float(t) for t in short_leg_times))))
        if np.any(~np.isfinite(self.short_leg_times)) or np.any((self.short_leg_times <= 0) | (self.short_leg_times >= 1)):
            raise ValueError('Optional short-leg times must be finite and in (0,1) seconds')

    def _candidate_ids(self, states, released, route, elapsed, position, velocity, now):
        ids = [int(i) for i in released if i not in route and self.failed_until.get(int(i), 0) <= now]
        ids.sort(key=lambda i: np.linalg.norm(states[i, :3] + states[i, 3:6] * (elapsed + 1.5) - position - velocity * 1.5))
        return ids[:self.branch_targets]

    def _node_priority(self, node, states, released, now):
        return sum(node[1])

    def _intercept_target(self, states, index, duration):
        return states[index, :3] + states[index, 3:6] * duration

    def _chain_curves(self, position, velocity, targets, durations, drift):
        return free_chain(position, velocity, self.acceleration, targets, durations, drift, self.terminal_weight)

    def _make_plan(self, observation, position, velocity, now):
        states = np.asarray(observation['balloon_states'], dtype=float)
        released = np.flatnonzero(np.asarray(observation['balloon_status']).reshape(-1) == 1)
        drift = states[:, 3:6]
        while self.route and self.route[0] not in released:
            self.route.pop(0)
            self.deadlines.pop(0)
            self.ends_v.pop(0)
            self.ends_a.pop(0)
        if self.route and (self.commitment or self.route[0] == self.target_index):
            duration = self.deadlines[0] - now
            if duration > 0.12:
                target = self._intercept_target(states, self.route[0], duration)
                curve = boundary_curve(position, velocity, self.acceleration, target, self.ends_v[0], self.ends_a[0], duration)
                if self._feasible(curve, duration, now):
                    self._select(self.route[0], curve, duration, now)
                    return
                if self.plan is not None and now < self.plan_start + self.plan_duration:
                    return
            else:
                self.failed_until[self.route[0]] = now + 1.0
        remaining = self.launch_time + self.burn_time - now - 0.0001
        initial_force = self.acceleration - G - self.disturbance
        limit = 0.98 * self.available_acceleration(now)
        if np.linalg.norm(initial_force) > limit and limit > 0:
            self.acceleration = G + self.disturbance + initial_force * limit / np.linalg.norm(initial_force)
        beam = [((), (), None, position, velocity)]
        best = None
        self.diagnostics['beam_searches'] += 1
        for depth in range(self.search_depth):
            children = []
            for route, times, _, end_p, end_v in beam:
                elapsed = sum(times)
                ids = self._candidate_ids(states, released, route, elapsed, end_p, end_v, now)
                ts = np.arange(1.0 if depth else 2.0, min(self.leg_horizon, remaining - elapsed) + 0.001, self.time_grid)
                if depth and len(self.short_leg_times):
                    short = self.short_leg_times[self.short_leg_times <= min(self.leg_horizon, remaining - elapsed)]
                    ts = np.r_[short, ts]
                for index in ids:
                    for t in ts:
                        ids2, durations = (route + (index,), times + (float(t),))
                        arrivals = np.cumsum(durations)
                        targets = states[list(ids2), :3] + drift[list(ids2)] * arrivals[:, None]
                        curves = self._chain_curves(position, velocity, targets, durations, drift[index])
                        starts = now + np.r_[0.0, arrivals[:-1]]
                        valid, ps, vs, acs = self._valid(curves, np.asarray(durations), starts[:, None], samples=17)
                        self.diagnostics['beam_candidates'] += 1
                        if not np.all(valid):
                            continue
                        self.diagnostics['beam_feasible'] += 1
                        children.append((ids2, durations, curves, ps[-1], vs[-1]))
            if not children:
                break
            children.sort(key=lambda n: self._node_priority(n, states, released, now))
            beam, counts = ([], {})
            for node in children:
                key = (node[0][0], node[0][-1])
                if counts.get(key, 0) >= 2:
                    continue
                counts[key] = counts.get(key, 0) + 1
                beam.append(node)
                if len(beam) >= self.beam_width:
                    break
            best = beam[0]
            self.diagnostics['maximum_depth'] = max(self.diagnostics['maximum_depth'], depth + 1)
        if best is None:
            return super()._make_plan(observation, position, velocity, now)
        route, times, curves, _, _ = best
        valid, _, vs, acs = self._valid(curves, np.asarray(times), (now + np.r_[0.0, np.cumsum(times)[:-1]])[:, None], samples=65)
        if not np.all(valid):
            return super()._make_plan(observation, position, velocity, now)
        self.route, self.deadlines = (list(route), list(now + np.cumsum(times)))
        self.ends_v, self.ends_a = (list(vs), list(acs))
        self.route_events.append([now, list(route), float(sum(times))])
        self._select(route[0], curves[0], times[0], now)

def launch_axes(states, released, position, available, count=3):
    """Constant-acceleration intercept estimates from observed released targets.

    These generate directions, not executable plans. Full chain feasibility is
    checked separately, including changing thrust-to-mass ratio and turn rate.
    """
    candidates = []
    for index in released:
        for duration in np.arange(2.0, 10.01, 0.5):
            target = states[index, :3] + states[index, 3:6] * duration
            accel = 2 * (target - position) / duration ** 2
            force = accel - G
            norm = np.linalg.norm(force)
            if norm <= 0.98 * available and accel[2] > 0.1 and (force[2] >= norm * np.cos(np.radians(35.0))):
                candidates.append((duration, force / norm))
                break
    axes = [np.array([0.0, 0.0, 1.0])]
    if count <= 0:
        return axes
    for _, axis in sorted(candidates, key=lambda item: item[0]):
        if all((np.dot(axis, other) < np.cos(np.radians(4.0)) for other in axes)):
            axes.append(axis)
        if len(axes) >= count + 1:
            break
    return axes

class ChainLaunchAgent(ChainBeamAgent):

    def __init__(self, given_parameters, launch_candidates=3, **kwargs):
        super().__init__(given_parameters, **kwargs)
        self.launch_candidates = int(launch_candidates)
        if self.launch_candidates < 0:
            raise ValueError('launch_candidates must be nonnegative')
        self.launch_comparisons = []
        self.launch_selected = False

    def get_action(self, observation):
        now = float(observation['simulation_time'])
        if not self.launched and (not self.launch_selected) and (now >= self.launch_time):
            states = np.asarray(observation['balloon_states'], dtype=float)
            released = np.flatnonzero(np.asarray(observation['balloon_status']).reshape(-1) == 1)
            position = np.array([0.0, 0.0, self.elevation])
            available = self.available_acceleration(now)
            original = copy.deepcopy(self.__dict__)
            best = None
            comparisons = []
            for axis in launch_axes(states, released, position, available, self.launch_candidates):
                self.__dict__ = copy.deepcopy(original)
                self.acceleration = G + available * axis
                self._make_plan(observation, position, np.zeros(3), now)
                count = len(self.route)
                finish = self.deadlines[-1] - now if count else float('inf')
                key = (-count, finish)
                comparisons.append([axis.tolist(), count, finish if count else None])
                if best is None or key < best[0]:
                    best = (key, axis.copy())
            self.__dict__ = original
            axis = best[1]
            inclination = np.degrees(np.arcsin(np.clip(axis[2], -1.0, 1.0)))
            heading = np.degrees(np.arctan2(axis[0], axis[1])) % 360.0
            self.launch_attitude = np.array([inclination, heading])
            self.quaternion = get_initial_attitude(inclination, heading)
            self.launch_selected = True
            self.launch_comparisons = comparisons
        return super().get_action(observation)

class FinalApproachMixin:

    def __init__(self, given_parameters, final_approach_guard=0.15, **kwargs):
        super().__init__(given_parameters, **kwargs)
        self.final_approach_guard = float(final_approach_guard)
        if not np.isfinite(self.final_approach_guard) or self.final_approach_guard < 0:
            raise ValueError('Nonnegative finite final approach guard required')
        self.diagnostics['final_approach_holds'] = 0

    def _make_plan(self, observation, position, velocity, now):
        if self.plan is not None and self.target_index is not None:
            remaining = self.plan_start + self.plan_duration - now
            status = np.asarray(observation['balloon_status']).reshape(-1)
            if 0 < remaining <= self.final_approach_guard and status[self.target_index] == 1:
                self.diagnostics['final_approach_holds'] += 1
                return
        return super()._make_plan(observation, position, velocity, now)

    def get_action(self, observation):
        action = super().get_action(observation)
        if self.plan is not None and self.target_index is not None:
            now = float(observation['simulation_time'])
            deadline = self.plan_start + self.plan_duration
            if 0 < deadline - now <= self.final_approach_guard:
                self.next_plan = min(self.next_plan, deadline)
        return action

class FinalApproachAgent(FinalApproachMixin, ChainLaunchAgent):
    pass

def constraint_margins(agent, curves, durations, now, samples=25):
    """Nonnegative means feasible in the same model as MomentumBeamAgent."""
    durations = np.asarray(durations)
    p, _, a, j = batch_samples(curves, durations, samples)
    times = now + np.r_[0.0, np.cumsum(durations)[:-1]][:, None] + durations[:, None] * np.linspace(0, 1, samples)
    limit = agent.thrust / np.maximum(agent.dry_mass, agent.initial_mass - agent.mass_flow * np.maximum(times - agent.launch_time, 0.0))
    force = a - G - agent.disturbance
    magnitude = np.maximum(np.linalg.norm(force, axis=2), 1e-09)
    axis = force / magnitude[:, :, None]
    transverse = j - axis * np.sum(axis * j, axis=2)[:, :, None]
    rate = np.linalg.norm(transverse, axis=2) / magnitude
    throttle = magnitude / limit
    slew = np.diff(throttle, axis=1) / np.diff(times, axis=1)
    reserve = agent.reserve * np.linspace(0.0, 1.0, samples)
    return np.r_[((limit - reserve - magnitude) / 12.0).ravel(), ((magnitude - 0.5) / 12.0).ravel(), (axis[:, :, 2] - np.cos(agent.max_tilt)).ravel(), ((agent.max_axis_rate - rate) / max(agent.max_axis_rate, 0.1)).ravel(), ((p[:, :, 2] - agent.elevation + 0.05) / 10.0).ravel(), ((agent.control['throttle_rate_limit'] - np.abs(slew)) / max(agent.control['throttle_rate_limit'], 0.1)).ravel()]

def repair_margins(agent, curves, durations, now, samples, safety):
    margins = constraint_margins(agent, curves, durations, now, samples)
    if safety:
        arrivals = np.cumsum(durations)
        elapsed = np.r_[0.0, arrivals[:-1]][:, None] + np.asarray(durations)[:, None] * np.linspace(0, 1, samples)
        guard = safety * (elapsed / arrivals[-1]).ravel()
        size = len(guard)
        for block in (0, 2, 3):
            margins[block * size:(block + 1) * size] -= guard
    return margins

def closest_approaches(curves, durations, states, samples=33):
    """Piecewise-linear relative-motion proximity; a planning approximation."""
    positions, _, _, _ = batch_samples(curves, durations, samples)
    ts = np.r_[0.0, np.cumsum(durations)[:-1]][:, None] + np.asarray(durations)[:, None] * np.linspace(0, 1, samples)
    points, times = (positions.reshape(-1, 3), ts.reshape(-1))
    relative = points[:, None, :] - states[None, :, :3] - times[:, None, None] * states[None, :, 3:6]
    delta = np.diff(relative, axis=0)
    alpha = np.clip(-np.sum(relative[:-1] * delta, axis=2) / np.maximum(np.sum(delta * delta, axis=2), 1e-12), 0.0, 1.0)
    distance = np.linalg.norm(relative[:-1] + alpha[:, :, None] * delta, axis=2)
    closest = np.argmin(distance, axis=0)
    columns = np.arange(len(states))
    return (distance[closest, columns], times[closest] + alpha[closest, columns] * np.diff(times)[closest])

def optimize_times(agent, route, initial_times, states, position, velocity, now, iterations=20, safety=0.0):
    """Return dense-checked (durations, curves, end_v, end_a), or None.

    Times are independent decision variables, bounded by the remaining burn.
    Moving target positions are recomputed at every trial's arrival times.
    Inputs and agent state are never modified; solver success alone is not
    accepted as evidence of physical feasibility.
    """
    nominal = np.asarray(initial_times, dtype=float)
    remaining = agent.launch_time + agent.burn_time - now - 0.0001
    if not len(route) or len(route) != len(nominal) or (not np.all(np.isfinite(nominal))) or (np.min(nominal) < 0.3) or (sum(nominal) > remaining) or (not np.all(np.isfinite(states[list(route)]))):
        return None

    def curves(times):
        arrivals = np.cumsum(times)
        centers = states[list(route), :3] + states[list(route), 3:6] * arrivals[:, None]
        return free_chain(position, velocity, agent.acceleration, centers, times, states[route[-1], 3:6], agent.terminal_weight)

    def objective(times):
        return float(sum(times) + 0.03 * np.sum(((times - nominal) / nominal) ** 2))

    def margins(times, samples):
        return np.r_[remaining - sum(times), repair_margins(agent, curves(times), times, now, samples, safety)]
    x = nominal.copy()
    best = None
    for samples, budget in ((17, iterations), (65, max(6, iterations // 2))):
        try:
            candidate = curves(x)
            starts = (now + np.r_[0.0, np.cumsum(x)[:-1]])[:, None]
            valid, _, vs, acs = agent._valid(candidate, x, starts, samples=65)
            if np.all(valid) and np.min(margins(x, 65)) >= -1e-08:
                if best is None or objective(x) < objective(best[0]):
                    best = (x.copy(), candidate, vs, acs)
            result = minimize(objective, x, method='SLSQP', bounds=[(max(0.3, 0.6 * t), min(agent.leg_horizon, 1.5 * t)) for t in nominal], constraints={'type': 'ineq', 'fun': lambda t: margins(t, samples)}, options={'maxiter': budget, 'ftol': 1e-05})
            if not np.all(np.isfinite(result.x)):
                break
            x = result.x
        except (ValueError, np.linalg.LinAlgError, FloatingPointError):
            return best
    try:
        candidate = curves(x)
        starts = (now + np.r_[0.0, np.cumsum(x)[:-1]])[:, None]
        valid, _, vs, acs = agent._valid(candidate, x, starts, samples=65)
        if np.all(valid) and np.min(margins(x, 65)) >= -1e-08 and (best is None or objective(x) < objective(best[0])):
            best = (x.copy(), candidate, vs, acs)
    except (ValueError, np.linalg.LinAlgError, FloatingPointError):
        pass
    return best

class TimeAllocationAgent(FinalApproachAgent):

    def __init__(self, given_parameters, timing_candidates=4, timing_iterations=20, timing_safety=0.0, timing_only_more_hits=True, **kwargs):
        super().__init__(given_parameters, **kwargs)
        self.timing_candidates = int(timing_candidates)
        self.timing_iterations = int(timing_iterations)
        self.timing_safety = float(timing_safety)
        self.timing_only_more_hits = bool(timing_only_more_hits)
        if self.timing_candidates < 0 or self.timing_iterations < 1 or (not np.isfinite(self.timing_safety)) or (self.timing_safety < 0):
            raise ValueError('Invalid time-allocation settings')
        self.diagnostics.update(timing_solves=0, timing_adopted=0, timing_extra_planned_hits=0)

    def _make_plan(self, observation, position, velocity, now):
        events = len(self.route_events)
        super()._make_plan(observation, position, velocity, now)
        if not self.timing_candidates or not self.route or len(self.route_events) == events:
            return
        route = tuple(self.route)
        times = np.diff(np.r_[now, self.deadlines])
        if min(times) < 0.3:
            return
        states = np.asarray(observation['balloon_states'], dtype=float)
        released = np.flatnonzero(np.asarray(observation['balloon_status']).reshape(-1) == 1)
        if not set(route).issubset(set(released)):
            return

        def solve(ids, ts):
            self.diagnostics['timing_solves'] += 1
            return optimize_times(self, ids, ts, states, position, velocity, now, self.timing_iterations, self.timing_safety)
        base = solve(route, times)
        if base is None:
            return
        shorter, curves, _, _ = base
        best = (route, base) if not self.timing_only_more_hits and sum(shorter) < sum(times) - 0.05 else None
        distances, near_times = closest_approaches(curves, shorter, states)
        arrivals = np.cumsum(shorter)
        candidates = [int(i) for i in released if i not in route and self.failed_until.get(int(i), 0) <= now and (distances[i] < 25.0) and (near_times[i] >= 0.3) and (np.min(np.abs(arrivals - near_times[i])) >= 0.3)]
        candidates.sort(key=lambda i: distances[i])
        for index in candidates[:self.timing_candidates]:
            slot = int(np.searchsorted(arrivals, near_times[index]))
            proposal = route[:slot] + (index,) + route[slot:]
            ts = np.diff(np.r_[0.0, np.insert(arrivals, slot, near_times[index])])
            result = solve(proposal, ts)
            if result is not None and (best is None or (-len(proposal), sum(result[0])) < (-len(best[0]), sum(best[1][0]))):
                best = (proposal, result)
        if best is None:
            return
        ids, (ts, curves, vs, acs) = best
        self.route, self.deadlines = (list(ids), list(now + np.cumsum(ts)))
        self.ends_v, self.ends_a = (list(vs), list(acs))
        self.route_events[-1] = [now, list(ids), float(sum(ts))]
        self._select(ids[0], curves[0], ts[0], now)
        self.diagnostics['timing_adopted'] += 1
        self.diagnostics['timing_extra_planned_hits'] += len(ids) - len(route)

class ChainSubmissionAgent(TimeAllocationAgent):
    pass

