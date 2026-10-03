"""Observation-only first-leg shadow rollout for Scenario 4 experiments.

This is a *screening model*, not the official 6-DOF simulator.  It propagates
the current filtered navigation/attitude state through the existing spline
tracking law, public thrust/mass parameters, and published actuator limits.
Aerodynamic moments, changing inertia, and future wind are not modeled;
``disturbance`` is a constant acceleration estimated from observations.
Nothing here reads a seed, simulator state, replay, or future balloon data.
"""

from dataclasses import dataclass

import numpy as np


_GRAVITY = np.array([0.0, 0.0, -9.80665])


@dataclass(frozen=True)
class ShadowResult:
    """Predicted miss and tracking health for one candidate first leg.

    The three-sigma bounds describe *pointwise* position uncertainty at the
    predicted closest time, not the probability of a hit anywhere on the path.
    """

    closest_distance: float
    closest_time: float
    final_position: tuple[float, float, float]
    final_velocity: tuple[float, float, float]
    nominal_closest_distance: float
    capture_radius: float
    three_sigma_distance_lower: float
    three_sigma_distance_upper: float
    max_reference_error: float
    max_attitude_error_degrees: float
    acceleration_saturated_steps: int
    gimbal_limited_steps: int
    throttle_limited_steps: int
    burnout_steps: int
    steps: int

    @property
    def nominal_capture(self):
        return self.closest_distance <= self.capture_radius

    @property
    def robust_capture(self):
        return self.three_sigma_distance_upper <= self.capture_radius


def _vector(value, length, name):
    array = np.asarray(value, dtype=float)
    if array.shape != (length,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite vector of length {length}")
    return array.copy()


def _multiply_quaternions(left, right):
    w1, x1, y1, z1 = left
    w2, x2, y2, z2 = right
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def _rotate_body_to_world(quaternion, vector):
    qv = quaternion[1:]
    cross = 2.0 * np.cross(qv, vector)
    return vector + quaternion[0] * cross + np.cross(qv, cross)


def _rotate_world_to_body(quaternion, vector):
    conjugate = quaternion * np.array([1.0, -1.0, -1.0, -1.0])
    return _rotate_body_to_world(conjugate, vector)


def _sample_curve(curve, duration, elapsed):
    s = elapsed / duration
    powers = s ** np.arange(6)
    position = powers @ curve
    velocity = (np.arange(1, 6) * s ** np.arange(5)) @ curve[1:] / duration
    acceleration = (
        (np.arange(2, 6) * np.arange(1, 5) * s ** np.arange(4))
        @ curve[2:] / duration**2
    )
    jerk = (
        (np.arange(3, 6) * np.arange(2, 5) * np.arange(1, 4)
         * s ** np.arange(3)) @ curve[3:] / duration**3
    )
    return position, velocity, acceleration, jerk


def _allocate_acceleration(requested, available, max_tilt):
    allocated = requested.copy()
    allocated[2] = np.clip(allocated[2], 0.0, available)
    lateral_limit = min(
        np.sqrt(max(0.0, available**2 - allocated[2]**2)),
        allocated[2] * np.tan(max_tilt),
    )
    lateral = np.linalg.norm(allocated[:2])
    if lateral > lateral_limit:
        allocated[:2] *= lateral_limit / lateral
    return allocated


def _advance_actuator(desired, previous, lag, rate_limit, timestep, lower, upper):
    """Mirror the agent's lag-inverting command and public actuator limits."""
    limited_desired = float(np.clip(desired, lower, upper))
    max_change = rate_limit * timestep
    desired_output = previous + np.clip(
        limited_desired - previous, -max_change, max_change)
    alpha = 1.0 if lag in (None, 0) else timestep / (float(lag) + timestep)
    raw_command = (desired_output - (1.0 - alpha) * previous) / alpha
    command = float(np.clip(raw_command, lower, upper))
    filtered = alpha * command + (1.0 - alpha) * previous
    output = previous + np.clip(filtered - previous, -max_change, max_change)
    limited = (abs(desired - limited_desired) > 1e-9 or
               abs(limited_desired - previous) > max_change + 1e-9 or
               abs(raw_command - command) > 1e-9)
    return float(np.clip(output, lower, upper)), bool(limited)


def _segment_closest(first_start, first_end, second_start, second_end):
    relative_start = first_start - second_start
    relative_change = (first_end - first_start) - (second_end - second_start)
    denominator = float(np.dot(relative_change, relative_change))
    alpha = (float(np.clip(-np.dot(relative_start, relative_change) /
                           denominator, 0.0, 1.0))
             if denominator > 1e-12 else 0.0)
    return float(np.linalg.norm(relative_start + alpha * relative_change)), alpha


def predict_first_leg(
    *, given_parameters, curve, duration, now, launch_time,
    position, velocity, quaternion, angular_velocity,
    gimbal_output, throttle_output, target_position, target_velocity,
    disturbance=(0.0, 0.0, 0.0), roll_output=0.0,
    tracking_frequency=1.3, attitude_frequency=4.0,
    max_tilt_degrees=75.0, max_axis_rate=1.2,
    step=None, initial_position_sigma=0.0, target_position_sigma=0.0,
    target_velocity_sigma=0.0, model_acceleration_sigma=0.0,
):
    """Roll out the current first-leg spline and a linear target forecast.

    All initial states must come from the agent's filtered observation and
    actuator estimate.  ``curve`` is the agent's normalized-time quintic
    reference.  Uncertainty inputs are optional isotropic 1-sigma estimates
    from *observed* residuals; zeros mean no uncertainty estimate, not certainty.
    """
    curve = np.asarray(curve, dtype=float)
    if curve.shape != (6, 3) or not np.all(np.isfinite(curve)):
        raise ValueError("curve must be finite with shape (6, 3)")
    position = _vector(position, 3, "position")
    velocity = _vector(velocity, 3, "velocity")
    quaternion = _vector(quaternion, 4, "quaternion")
    angular_velocity = _vector(angular_velocity, 3, "angular_velocity")
    gimbal = _vector(gimbal_output, 2, "gimbal_output")
    target_position = _vector(target_position, 3, "target_position")
    target_velocity = _vector(target_velocity, 3, "target_velocity")
    disturbance = _vector(disturbance, 3, "disturbance")
    norm = np.linalg.norm(quaternion)
    if norm <= 0:
        raise ValueError("quaternion must be nonzero")
    quaternion /= norm

    step = (float(given_parameters["simulation"]["time_step"])
            if step is None else float(step))
    scalars = (duration, now, launch_time, throttle_output, roll_output, step,
               tracking_frequency, attitude_frequency, max_tilt_degrees,
               max_axis_rate, initial_position_sigma, target_position_sigma,
               target_velocity_sigma, model_acceleration_sigma)
    if (not np.all(np.isfinite(scalars)) or duration <= 0 or step <= 0 or
            tracking_frequency <= 0 or attitude_frequency <= 0 or
            not 0 < max_tilt_degrees < 90 or max_axis_rate <= 0 or
            min(initial_position_sigma, target_position_sigma,
                target_velocity_sigma, model_acceleration_sigma) < 0):
        raise ValueError("invalid shadow-rollout settings")

    rocket = given_parameters["rocket"]
    control = rocket["control"]
    body, motor, tank = (rocket["rocket_body"], rocket["motor"], rocket["tank"])
    grain_mass = (np.pi * (motor["grain_outer_radius"]**2 -
                           motor["grain_initial_inner_radius"]**2) *
                  motor["grain_initial_height"] * motor["grain_density"] *
                  motor["grain_number"])
    dry_mass = float(body["mass"] + motor["dry_mass"])
    initial_mass = (dry_mass + grain_mass + tank["initial_liquid_mass"] +
                    tank["initial_gas_mass"])
    burn_time = float(motor["burn_time"])
    mass_flow = float(tank["liquid_mass_flow_rate_out"] + grain_mass / burn_time)
    thrust = float(motor["thrust_source"])
    inertia = np.asarray(body["inertia"], dtype=float).copy()
    tank_z = tank["tank_position"] + motor["motor_position"]
    inertia[:2] += (tank["initial_liquid_mass"] + tank["initial_gas_mass"]) * tank_z**2
    lever = max(abs(motor["motor_position"] + motor["nozzle_position"]), 0.1)
    radius = float(given_parameters["balloon"]["radius"])
    if (min(dry_mass, initial_mass, burn_time, thrust, lever, radius) <= 0 or
            np.any(inertia <= 0) or not np.all(np.isfinite(inertia))):
        raise ValueError("invalid public rocket parameters")
    max_gimbal = float(control["max_gimbal_angle"])
    throttle_lower, throttle_upper = map(float, control["throttle_range"])
    if (np.any(np.abs(gimbal) > max_gimbal + 1e-9) or
            not throttle_lower <= throttle_output <= throttle_upper):
        raise ValueError("initial actuator output outside public limits")

    steps = int(np.ceil(duration / step))
    nominal_previous = _sample_curve(curve, duration, 0.0)[0]
    rocket_previous = position.copy()
    balloon_previous = target_position.copy()
    closest_distance = float(np.linalg.norm(position - target_position))
    nominal_closest = float(np.linalg.norm(nominal_previous - target_position))
    closest_time = 0.0
    max_reference_error = 0.0
    max_attitude_error = 0.0
    acceleration_saturated_steps = 0
    gimbal_limited_steps = 0
    throttle_limited_steps = 0
    burnout_steps = 0
    max_tilt = np.radians(max_tilt_degrees)

    for index in range(steps):
        elapsed = min(index * step, duration)
        next_elapsed = min((index + 1) * step, duration)
        dt = next_elapsed - elapsed
        if dt <= 0:
            break
        age = max(0.0, now + elapsed - launch_time)
        available = (thrust / max(dry_mass, initial_mass - mass_flow * age)
                     if age < burn_time else 0.0)
        if available == 0:
            burnout_steps += 1
        reference_p, reference_v, reference_a, reference_jerk = _sample_curve(
            curve, duration, elapsed)
        wn = tracking_frequency
        desired_acceleration = (reference_a + wn**2 * (reference_p - position) +
                                2.0 * wn * (reference_v - velocity))
        requested = desired_acceleration - _GRAVITY - disturbance
        allocated = _allocate_acceleration(requested, available, max_tilt)
        acceleration_saturated_steps += int(np.linalg.norm(allocated - requested) > 0.1)
        magnitude = float(np.linalg.norm(allocated))
        axis = (allocated / magnitude if magnitude > 1e-9 else
                np.array([0.0, 0.0, 1.0]))
        body_axis = _rotate_body_to_world(quaternion, np.array([0.0, 0.0, 1.0]))
        angle = np.arccos(np.clip(np.dot(body_axis, axis), -1.0, 1.0))
        max_attitude_error = max(max_attitude_error, float(np.degrees(angle)))
        error = _rotate_world_to_body(quaternion, np.cross(body_axis, axis))
        omega_world = np.cross(axis, reference_jerk / max(magnitude, 1.0))
        omega_reference = _rotate_world_to_body(quaternion, omega_world)
        omega_reference = np.clip(omega_reference, -max_axis_rate, max_axis_rate)
        desired_angular_acceleration = (attitude_frequency**2 * error +
                                        2.0 * attitude_frequency *
                                        (omega_reference - angular_velocity))
        effective_force = thrust * max(throttle_output, 0.05)
        torque = inertia * desired_angular_acceleration
        raw_gimbal = np.degrees(np.arcsin(np.clip(
            torque[:2] / (effective_force * lever), -1.0, 1.0)))
        for axis_index in range(2):
            gimbal[axis_index], limited = _advance_actuator(
                raw_gimbal[axis_index], gimbal[axis_index],
                control["gimbal_time_constant"], control["gimbal_rate_limit"],
                dt, -max_gimbal, max_gimbal)
            gimbal_limited_steps += int(limited)
        desired_throttle = magnitude / max(available, 1e-9)
        throttle_output, limited = _advance_actuator(
            desired_throttle, throttle_output, control["throttle_time_constant"],
            control["throttle_rate_limit"], dt,
            throttle_lower, throttle_upper)
        throttle_limited_steps += int(limited)

        desired_roll = float(np.clip(
            -2.0 * attitude_frequency * inertia[2] * angular_velocity[2],
            -control["max_roll_torque"], control["max_roll_torque"]))
        roll_output, _ = _advance_actuator(
            desired_roll, roll_output, control["roll_torque_time_constant"],
            control["torque_rate_limit"], dt,
            -control["max_roll_torque"], control["max_roll_torque"])
        # The official 6-DOF dynamics also include aerodynamic moments and
        # time-varying inertia; these public-limit dynamics intentionally do not.
        angular_acceleration = np.array([
            np.sin(np.radians(gimbal[0])) * thrust * throttle_output * lever / inertia[0],
            np.sin(np.radians(gimbal[1])) * thrust * throttle_output * lever / inertia[1],
            roll_output / inertia[2],
        ])
        omega_before = angular_velocity.copy()
        angular_velocity += angular_acceleration * dt
        angular_increment = 0.5 * (omega_before + angular_velocity) * dt
        rotation = np.linalg.norm(angular_increment)
        if rotation > 1e-12:
            delta_q = np.r_[np.cos(rotation / 2.0),
                            np.sin(rotation / 2.0) * angular_increment / rotation]
            quaternion = _multiply_quaternions(quaternion, delta_q)
            quaternion /= np.linalg.norm(quaternion)

        world_acceleration = (_GRAVITY + disturbance +
                              available * throttle_output * body_axis)
        position += velocity * dt + 0.5 * world_acceleration * dt**2
        velocity += world_acceleration * dt
        balloon_next = target_position + target_velocity * next_elapsed
        nominal_next = _sample_curve(curve, duration, next_elapsed)[0]
        distance, fraction = _segment_closest(
            rocket_previous, position, balloon_previous, balloon_next)
        if distance < closest_distance:
            closest_distance = distance
            closest_time = elapsed + fraction * dt
        nominal_distance, _ = _segment_closest(
            nominal_previous, nominal_next, balloon_previous, balloon_next)
        nominal_closest = min(nominal_closest, nominal_distance)
        max_reference_error = max(
            max_reference_error, float(np.linalg.norm(position - nominal_next)))
        rocket_previous = position.copy()
        balloon_previous = balloon_next
        nominal_previous = nominal_next

    sigma = np.sqrt(initial_position_sigma**2 + target_position_sigma**2 +
                    (target_velocity_sigma * closest_time)**2 +
                    (0.5 * model_acceleration_sigma * closest_time**2)**2)
    return ShadowResult(
        closest_distance=closest_distance,
        closest_time=closest_time,
        final_position=tuple(float(value) for value in position),
        final_velocity=tuple(float(value) for value in velocity),
        nominal_closest_distance=nominal_closest,
        capture_radius=radius,
        three_sigma_distance_lower=max(0.0, closest_distance - 3.0 * sigma),
        three_sigma_distance_upper=closest_distance + 3.0 * sigma,
        max_reference_error=max_reference_error,
        max_attitude_error_degrees=max_attitude_error,
        acceleration_saturated_steps=acceleration_saturated_steps,
        gimbal_limited_steps=gimbal_limited_steps,
        throttle_limited_steps=throttle_limited_steps,
        burnout_steps=burnout_steps,
        steps=steps,
    )
