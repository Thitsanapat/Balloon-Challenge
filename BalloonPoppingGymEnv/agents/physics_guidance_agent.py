"""Observation-only polynomial interception with thrust and turn constraints.

Derivation, sources, assumptions, and limitations: doc/physics_guidance.md.
This is a sampled feasible-trajectory search, not a full SCvx/MPC solver.
"""

import numpy as np

from BalloonPoppingGymEnv.agents.base_agent import BaseAgent
from BalloonPoppingGymEnv.agents.multi_target_agent import (
    _multiply_quaternions,
    _rotate_body_to_world,
    _rotate_world_to_body,
)
from BalloonPoppingGymEnv.envs.balloon_world import get_initial_attitude


G = np.array([0.0, 0.0, -9.80665])


def quintic_intercept(position, velocity, acceleration, target, target_velocity, duration):
    """Minimum integrated squared jerk for fixed endpoint position/velocity/accel.

    Coefficients use normalized time s=t/duration, avoiding small/large-time
    Vandermonde systems. Terminal acceleration is zero (constant balloon drift).
    """
    t = float(duration)
    if not np.isfinite(t) or t <= 0:
        raise ValueError("duration must be finite and positive")
    c0 = np.asarray(position, dtype=float)
    c1 = np.asarray(velocity, dtype=float) * t
    c2 = np.asarray(acceleration, dtype=float) * t * t / 2
    dp = np.asarray(target, dtype=float) - c0 - c1 - c2
    dv = np.asarray(target_velocity, dtype=float) * t - c1 - 2 * c2
    da = -2 * c2
    return np.stack((c0, c1, c2, 10*dp-4*dv+da/2,
                     -15*dp+7*dv-da, 6*dp-3*dv+da/2))


def sample_curve(coefficients, duration, elapsed):
    s = np.asarray(elapsed, dtype=float) / duration
    values = []
    c = coefficients.copy()
    for derivative in range(4):
        powers = s[..., None] ** np.arange(len(c))
        values.append((powers @ c) / duration**derivative)
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
    lateral_limit = min(
        np.sqrt(max(0.0, available**2-command[2]**2)),
        command[2] * np.tan(max_tilt),
    )
    lateral = np.linalg.norm(command[:2])
    if lateral > lateral_limit:
        command[:2] *= lateral_limit / lateral
    return command


class PhysicsGuidanceAgent(BaseAgent):
    """Replan smooth intercepts; infer all feedback from permitted sensors."""

    def __init__(self, given_parameters, launch_time=4.0, replan_interval=0.4,
                 max_tilt=55.0, max_axis_rate=0.7, attitude_frequency=4.0,
                 tracking_frequency=1.0, horizon=12.0, arrival_speed=0.0,
                 lookahead_velocity=False, minimum_vertical_acceleration=None,
                 direct_rate_gain=None, force_full_throttle=False):
        super().__init__(given_parameters)
        self.dt = float(given_parameters["simulation"]["time_step"])
        self.elevation = float(given_parameters["environment"]["elevation"])
        self.radius = float(given_parameters["balloon"]["radius"])
        rocket = given_parameters["rocket"]
        self.control = rocket["control"]
        body, motor, tank = rocket["rocket_body"], rocket["motor"], rocket["tank"]
        grain_mass = (np.pi * (motor["grain_outer_radius"]**2 -
                      motor["grain_initial_inner_radius"]**2) *
                      motor["grain_initial_height"] * motor["grain_density"] *
                      motor["grain_number"])
        self.dry_mass = float(body["mass"] + motor["dry_mass"])
        self.initial_mass = self.dry_mass + grain_mass + tank["initial_liquid_mass"] + tank["initial_gas_mass"]
        self.burn_time = float(motor["burn_time"])
        self.mass_flow = tank["liquid_mass_flow_rate_out"] + grain_mass / self.burn_time
        self.thrust = float(motor["thrust_source"])
        # Parallel-axis approximation; not access to the environment's motor.
        tank_z = tank["tank_position"] + motor["motor_position"]
        tank_mass = tank["initial_liquid_mass"] + tank["initial_gas_mass"]
        self.inertia = np.asarray(body["inertia"], dtype=float).copy()
        self.inertia[:2] += tank_mass * tank_z**2
        self.lever = max(abs(motor["motor_position"] + motor["nozzle_position"]), 0.1)
        self.launch_time = float(launch_time)
        self.replan_interval = float(replan_interval)
        self.max_tilt = np.radians(max_tilt)
        self.max_axis_rate = float(max_axis_rate)
        self.attitude_frequency = float(attitude_frequency)
        self.tracking_frequency = float(tracking_frequency)
        self.horizon = float(horizon)
        self.arrival_speed = float(arrival_speed)
        self.lookahead_velocity = bool(lookahead_velocity)
        self.minimum_vertical_acceleration = (None if minimum_vertical_acceleration is None
                                              else float(minimum_vertical_acceleration))
        self.direct_rate_gain = (None if direct_rate_gain is None
                                 else float(direct_rate_gain))
        self.force_full_throttle = bool(force_full_throttle)
        self.launched = False
        self.launch_attitude = np.array([90.0, 0.0])
        self.quaternion = np.asarray(get_initial_attitude(90.0, 0.0))
        self.previous_gyro = np.zeros(3)
        self.previous_velocity = None
        self.acceleration = np.zeros(3)
        self.disturbance = np.zeros(3)
        self.previous_tvc = np.zeros(2)
        self.previous_throttle = 1.0
        self.target_index = None
        self.target_events = []
        self.plan = None
        self.plan_start = 0.0
        self.plan_duration = 0.0
        self.next_plan = 0.0
        self.last_desired_axis = np.array([0.0, 0.0, 1.0])
        self.failed_until = {}
        self.diagnostics = {"plans": 0, "no_feasible_plan": 0, "saturated_steps": 0,
                            "position_error_sum": 0.0, "tracking_steps": 0}

    def available_acceleration(self, now):
        elapsed = max(0.0, now - self.launch_time)
        if elapsed >= self.burn_time:
            return 0.0
        mass = max(self.dry_mass, self.initial_mass-self.mass_flow*elapsed)
        return self.thrust/mass

    def _feasible(self, c, duration, now):
        times = np.linspace(0.0, duration, 33)
        p, v, a, jerk = sample_curve(c, duration, times)
        thrust_accel = a - G - self.disturbance
        magnitude = np.linalg.norm(thrust_accel, axis=1)
        limits = np.array([self.available_acceleration(now+t) for t in times])
        if np.any(magnitude > limits) or np.any(magnitude < 0.5):
            return False
        if np.any(thrust_accel[:, 2] < magnitude*np.cos(self.max_tilt)):
            return False
        axes = thrust_accel / magnitude[:, None]
        axis_rate = np.linalg.norm(jerk-axes*np.sum(axes*jerk, axis=1)[:, None], axis=1)/magnitude
        if np.any(axis_rate > self.max_axis_rate):
            return False
        throttle = magnitude/np.maximum(limits, 1e-9)
        if np.any(np.abs(np.diff(throttle)/np.diff(times)) > self.control["throttle_rate_limit"]):
            return False
        # A small launch tolerance accounts for ENU ground curvature.
        return bool(np.all(p[:, 2] >= self.elevation-0.05))

    def _make_plan(self, observation, position, velocity, now):
        states = np.asarray(observation["balloon_states"], dtype=float)
        status = np.asarray(observation["balloon_status"]).reshape(-1)
        released = np.flatnonzero(status == 1)
        moving = released[np.linalg.norm(states[released, 3:6], axis=1) > 0.5]
        shared_velocity = np.median(states[moving, 3:6], axis=0) if moving.size else np.zeros(3)
        candidates = [int(i) for i in released if self.failed_until.get(int(i), 0) <= now]
        locked = self.target_index in candidates
        if locked:
            candidates = [self.target_index]
        else:
            candidates.sort(key=lambda i: np.linalg.norm(states[i, :3]-position))
            candidates = candidates[:12]
        earliest = max(0.4, self.plan_start+self.plan_duration-now) if locked else 0.8
        best = None
        for index in candidates:
            drift = states[index, 3:6]
            if np.linalg.norm(drift) <= 0.5:
                drift = shared_velocity
            relative = states[index, :3]-position
            direction = relative/max(np.linalg.norm(relative), 1e-9)
            terminal_velocity = drift + self.arrival_speed*direction
            durations = np.unique(np.r_[earliest, np.arange(max(earliest, 0.8), self.horizon+0.01, 0.4)])
            for duration in durations:
                if best is not None and duration >= best[0]:
                    break
                target = states[index, :3] + drift*duration
                if self.lookahead_velocity and self.arrival_speed > 0:
                    others = released[released != index]
                    if others.size:
                        next_positions = states[others, :3]+states[others, 3:6]*duration
                        deltas = next_positions-target
                        distances = np.linalg.norm(deltas, axis=1)
                        nearest = int(np.argmin(distances))
                        # Permit braking/turning over half the inter-balloon distance:
                        # v^2 <= 2*a*(distance/2), with level-flight thrust reserve.
                        reserve = np.sqrt(max(0.0, self.available_acceleration(now+duration)**2-G[2]**2))
                        speed = min(self.arrival_speed, np.sqrt(reserve*distances[nearest]))
                        terminal_velocity = drift+speed*deltas[nearest]/max(distances[nearest], 1e-9)
                c = quintic_intercept(position, velocity, self.acceleration,
                                      target, terminal_velocity, duration)
                if self._feasible(c, duration, now):
                    best = (duration, index, c)
                    break
        if best is None:
            self.diagnostics["no_feasible_plan"] += 1
            # Keep executing a still-current plan before trying another target.
            if self.plan is not None and now < self.plan_start+self.plan_duration:
                return
            if self.target_index is not None:
                self.failed_until[self.target_index] = now + 2*self.replan_interval
            self.target_index = None
            self.plan = None
            return
        duration, index, c = best
        if index != self.target_index:
            self.target_events.append((now, index))
        self.target_index = index
        self.plan, self.plan_start, self.plan_duration = c, now, duration
        self.diagnostics["plans"] += 1

    def _attitude_action(self, axis, jerk, magnitude, gyro, available):
        body_axis = _rotate_body_to_world(self.quaternion, [0.0, 0.0, 1.0])
        error = _rotate_world_to_body(self.quaternion, np.cross(body_axis, axis))
        # Differential-flatness feedforward: omega_world = b3 x db3/dt.
        omega_world = np.cross(axis, jerk/max(magnitude, 1.0))
        omega_ref = _rotate_world_to_body(self.quaternion, omega_world)
        omega_ref = np.clip(omega_ref, -self.max_axis_rate, self.max_axis_rate)
        wn = self.attitude_frequency
        if self.direct_rate_gain is None:
            angular_accel = wn**2*error + 2*wn*(omega_ref-gyro)
            effective_force = self.thrust*max(self.previous_throttle, 0.05)
            torque = self.inertia*angular_accel
            tvc = np.degrees(np.arcsin(np.clip(
                torque[:2]/(effective_force*self.lever), -1.0, 1.0,
            )))
        else:
            desired_rate = np.clip(
                omega_ref[:2]+wn*error[:2],
                -self.max_axis_rate,
                self.max_axis_rate,
            )
            tvc = self.direct_rate_gain*(desired_rate-gyro[:2])
        tvc = np.clip(tvc, -self.control["max_gimbal_angle"], self.control["max_gimbal_angle"])
        change = self.control["gimbal_rate_limit"]*self.dt
        tvc = self.previous_tvc + np.clip(tvc-self.previous_tvc, -change, change)
        # Magnitude allocation is based on the feasible reference force.
        throttle = 1.0 if self.force_full_throttle else magnitude/max(available, 1e-9)
        throttle = np.clip(throttle, *self.control["throttle_range"])
        change = self.control["throttle_rate_limit"]*self.dt
        throttle = self.previous_throttle + np.clip(throttle-self.previous_throttle, -change, change)
        roll = float(np.clip(-2*wn*self.inertia[2]*gyro[2], -self.control["max_roll_torque"], self.control["max_roll_torque"]))
        self.previous_tvc = tvc.copy()
        self.previous_throttle = float(throttle)
        return tvc, roll, float(throttle)

    def get_action(self, observation):
        now = float(observation["simulation_time"])
        if not self.launched and now >= self.launch_time:
            self.launched = True
            self.launch_time = now + self.dt  # actual environment ignition step
        sensors = np.asarray(observation["rocket_sensors"], dtype=float)
        tvc, roll, throttle = np.zeros(2), 0.0, 1.0
        if self.launched and np.all(np.isfinite(sensors)):
            gyro = sensors[:3]
            increment = (gyro+self.previous_gyro)*self.dt/2
            angle = np.linalg.norm(increment)
            if angle > 1e-12:
                dq = np.r_[np.cos(angle/2), np.sin(angle/2)*increment/angle]
                self.quaternion = _multiply_quaternions(self.quaternion, dq)
                self.quaternion /= np.linalg.norm(self.quaternion)
            self.previous_gyro = gyro.copy()
            position, velocity = sensors[6:9], sensors[9:12]
            available = self.available_acceleration(now)
            if self.previous_velocity is not None:
                measured = (velocity-self.previous_velocity)/self.dt
                self.acceleration += self.dt/(0.08+self.dt)*(measured-self.acceleration)
                body_axis = _rotate_body_to_world(self.quaternion, [0, 0, 1])
                residual = measured-G-available*self.previous_throttle*body_axis
                # Bounded disturbance model includes drag and mass/pressure error.
                self.disturbance += self.dt/(0.8+self.dt)*(np.clip(residual, -3, 3)-self.disturbance)
            else:
                # The first post-launch sample has no velocity difference yet.
                # Seed the translational model from the actual launch attitude;
                # assuming vertical here makes an inclined launch immediately
                # command a contradictory pitch transient.
                body_axis = _rotate_body_to_world(self.quaternion, [0, 0, 1])
                self.acceleration = G + available*body_axis
            self.previous_velocity = velocity.copy()
            status = np.asarray(observation["balloon_status"]).reshape(-1)
            if self.target_index is not None and status[self.target_index] != 1:
                self.target_index = None
                self.plan = None
                self.next_plan = now
            if now >= self.next_plan:
                self._make_plan(observation, position, velocity, now)
                self.next_plan = now + self.replan_interval
            if self.plan is not None:
                elapsed = np.clip(now-self.plan_start, 0, self.plan_duration)
                p, v, a, jerk = sample_curve(self.plan, self.plan_duration, elapsed)
                wn = self.tracking_frequency
                acceleration = a+wn**2*(p-position)+2*wn*(v-velocity)
                self.diagnostics["position_error_sum"] += float(np.linalg.norm(p-position))
                self.diagnostics["tracking_steps"] += 1
            else:
                # Arrest a descent using the finite vertical acceleration reserve.
                acceleration = np.array([0.0, 0.0, max(0.0, -velocity[2])])
                jerk = np.zeros(3)
            requested = acceleration-G-self.disturbance
            if self.minimum_vertical_acceleration is not None and available > 0:
                # Energy-management mode: keep building vertical kinetic and
                # potential energy throughout powered flight. Horizontal
                # tracking receives only the thrust left after this support.
                requested[2] = max(requested[2], -G[2]+self.minimum_vertical_acceleration)
            allocated = allocate_acceleration(requested, available, self.max_tilt)
            if np.linalg.norm(allocated-requested) > 0.1:
                self.diagnostics["saturated_steps"] += 1
            magnitude = np.linalg.norm(allocated)
            axis = allocated/max(magnitude, 1e-9) if magnitude > 1e-9 else np.array([0., 0., 1.])
            self.last_desired_axis = axis.copy()
            tvc, roll, throttle = self._attitude_action(axis, jerk, magnitude, gyro, available)
        return {"launch": self.launched, "launch_inclination_heading": self.launch_attitude.copy(),
                "tvc": tvc, "roll": roll, "throttle": throttle}
