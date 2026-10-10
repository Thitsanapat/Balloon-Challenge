"""Scenario 4 experiment: compensate observed acceleration residuals.

The official environment requests an accelerometer with
``consider_gravity=True``, so the body-frame observation is specific force:
inertial acceleration minus gravity. After rotating it into world coordinates,
subtract the thrust predicted from public rocket parameters and the prior
throttle output. The remaining acceleration approximates wind, drag, and model
error. This agent uses only the observation and published given parameters.
"""

import numpy as np

from BalloonPoppingGymEnv.agents.scenario4_fusion_agent import Scenario4FusionAgent
from BalloonPoppingGymEnv.agents.submission_sqp_v2 import (
    _multiply_quaternions,
    _rotate_body_to_world,
)


def filter_acceleration_residual(previous, observed_world, thrust_axis_world,
                                 available, throttle, dt, time_constant, limit):
    """Bound and low-pass the non-thrust part of observed specific force."""
    previous = np.asarray(previous, dtype=float)
    observed_world = np.asarray(observed_world, dtype=float)
    thrust_axis_world = np.asarray(thrust_axis_world, dtype=float)
    residual = observed_world - float(available) * float(throttle) * thrust_axis_world
    bounded = np.clip(residual, -limit, limit)
    gain = -np.expm1(-dt / time_constant)
    return previous + gain * (bounded - previous), residual


class Scenario4DisturbanceAgent(Scenario4FusionAgent):
    """Add an IMU-based disturbance estimate to the existing fusion baseline."""

    def __init__(self, given_parameters, disturbance_time_constant=0.20,
                 disturbance_limit=3.0, **kwargs):
        super().__init__(given_parameters, **kwargs)
        self.disturbance_time_constant = float(disturbance_time_constant)
        self.disturbance_limit = float(disturbance_limit)
        if (not np.isfinite(self.disturbance_time_constant) or
                self.disturbance_time_constant <= 0 or
                not np.isfinite(self.disturbance_limit) or
                self.disturbance_limit <= 0):
            raise ValueError('Positive finite disturbance filter settings required')
        self.imu_disturbance = np.zeros(3)
        self.diagnostics['imu_disturbance_updates'] = 0
        self.diagnostics['imu_residual_clipped'] = 0

    def _filter_sensors(self, gyro, position, velocity):
        previous_time = self.previous_navigation_time
        filtered = super()._filter_sensors(gyro, position, velocity)
        now = self.observation_time
        if previous_time is None or not np.isfinite(now - previous_time) or now <= previous_time:
            return filtered

        elapsed = now - previous_time
        # This observation covers the interval before the parent integrates
        # its current gyro sample.  Use the midpoint attitude for acceleration.
        increment = (filtered[0] + self.previous_gyro) * elapsed / 2
        angle = float(np.linalg.norm(increment))
        if angle > 1e-12:
            half_turn = np.r_[np.cos(angle / 4),
                              np.sin(angle / 4) * increment / angle]
            midpoint_attitude = _multiply_quaternions(self.quaternion, half_turn)
        else:
            midpoint_attitude = self.quaternion
        observed_world = _rotate_body_to_world(
            midpoint_attitude, self.observed_acceleration)
        thrust_axis_world = _rotate_body_to_world(
            midpoint_attitude, [0., 0., 1.])
        self.imu_disturbance, raw_residual = filter_acceleration_residual(
            self.imu_disturbance, observed_world, thrust_axis_world,
            self.available_acceleration(now - elapsed), self.previous_throttle,
            elapsed, self.disturbance_time_constant, self.disturbance_limit)
        self.diagnostics['imu_disturbance_updates'] += 1
        self.diagnostics['imu_residual_clipped'] += int(
            np.any(np.abs(raw_residual) > self.disturbance_limit))
        # The inherited controller will still apply its small GNSS-derived
        # correction later in this step.  Keep this separate state so that
        # derivative noise does not accumulate in the IMU estimate.
        self.disturbance = self.imu_disturbance.copy()
        return filtered
