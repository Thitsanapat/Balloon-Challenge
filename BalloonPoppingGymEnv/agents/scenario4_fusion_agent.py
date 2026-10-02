"""Scenario 4 navigation experiment using only the published sensor observation.

RocketPy's accelerometer is created with ``consider_gravity=True`` in the
official environment, so its body-frame sample is specific force.  A
small position/velocity Kalman filter propagates that sample and corrects it
with the observed GNSS position and velocity.  Route search and flight control
remain those of TimeAllocationAgent.
"""

import numpy as np

from BalloonPoppingGymEnv.agents.submission_sqp_v2 import (
    G,
    _multiply_quaternions,
    _rotate_body_to_world,
)
from BalloonPoppingGymEnv.agents.time_allocation_agent import TimeAllocationAgent


def fuse_navigation_step(position, velocity, covariance, acceleration, measured_position,
                         measured_velocity, dt, position_sigma, velocity_sigma,
                         acceleration_sigma):
    """Predict and correct three independent [position, velocity] estimates."""
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError('Navigation interval must be finite and positive')
    position = np.asarray(position, dtype=float)
    velocity = np.asarray(velocity, dtype=float)
    acceleration = np.asarray(acceleration, dtype=float)
    measured_position = np.asarray(measured_position, dtype=float)
    measured_velocity = np.asarray(measured_velocity, dtype=float)
    covariance = np.asarray(covariance, dtype=float)
    position_sigma = np.broadcast_to(np.asarray(position_sigma, dtype=float), (3,))
    velocity_sigma = np.broadcast_to(np.asarray(velocity_sigma, dtype=float), (3,))
    if (any(x.shape != (3,) for x in (position, velocity, acceleration,
                                      measured_position, measured_velocity)) or
            covariance.shape != (3, 2, 2) or
            not np.all(np.isfinite(np.r_[position, velocity, acceleration,
                                        measured_position, measured_velocity,
                                        position_sigma, velocity_sigma])) or
            np.any(position_sigma < 0) or np.any(velocity_sigma < 0) or
            not np.isfinite(acceleration_sigma) or acceleration_sigma <= 0):
        raise ValueError('Invalid navigation state or sensor uncertainty')

    predicted_position = position + velocity * dt + 0.5 * acceleration * dt * dt
    predicted_velocity = velocity + acceleration * dt
    transition = np.array([[1., dt], [0., 1.]])
    noise_vector = np.array([0.5 * dt * dt, dt])
    process_noise = acceleration_sigma**2 * np.outer(noise_vector, noise_vector)
    result_position, result_velocity = np.empty(3), np.empty(3)
    result_covariance = np.empty_like(covariance)
    for axis in range(3):
        predicted_covariance = transition @ covariance[axis] @ transition.T + process_noise
        measurement_covariance = np.diag([max(position_sigma[axis]**2, 1e-12),
                                          max(velocity_sigma[axis]**2, 1e-12)])
        innovation_covariance = predicted_covariance + measurement_covariance
        gain = np.linalg.solve(innovation_covariance, predicted_covariance.T).T
        predicted = np.array([predicted_position[axis], predicted_velocity[axis]])
        measured = np.array([measured_position[axis], measured_velocity[axis]])
        corrected = predicted + gain @ (measured - predicted)
        residual_gain = np.eye(2) - gain
        posterior = (residual_gain @ predicted_covariance @ residual_gain.T +
                     gain @ measurement_covariance @ gain.T)
        result_position[axis], result_velocity[axis] = corrected
        result_covariance[axis] = (posterior + posterior.T) / 2
    return result_position, result_velocity, result_covariance


class Scenario4FusionAgent(TimeAllocationAgent):
    """Use accelerometer prediction and noise-weighted GNSS correction."""

    def __init__(self, given_parameters, process_acceleration_sigma=0.8, **kwargs):
        super().__init__(given_parameters, **kwargs)
        self.process_acceleration_sigma = float(process_acceleration_sigma)
        if not np.isfinite(self.process_acceleration_sigma) or self.process_acceleration_sigma <= 0:
            raise ValueError('Positive finite process acceleration sigma required')
        sensors = given_parameters['rocket']['sensors']
        self.gnss_position_sigma = np.array([
            sensors['gnss_position_accuracy'], sensors['gnss_position_accuracy'],
            sensors['gnss_altitude_accuracy']], dtype=float)
        self.gnss_velocity_sigma = np.full(3, float(sensors['gnss_velocity_accuracy']))
        self.accelerometer_noise_density = float(sensors['accelerometer_noise_density'])
        self.accelerometer_random_walk_density = float(sensors['accelerometer_random_walk_density'])
        self.navigation_covariance = np.zeros((3, 2, 2))
        self.previous_navigation_time = None
        self.observed_acceleration = None
        self.observation_time = None

    def get_action(self, observation):
        sensors = np.asarray(observation['rocket_sensors'], dtype=float)
        self.observed_acceleration = sensors[3:6].copy()
        self.observation_time = float(observation['simulation_time'])
        return super().get_action(observation)

    def _filter_sensors(self, gyro, position, velocity):
        gyro = np.asarray(gyro, dtype=float)
        position = np.asarray(position, dtype=float)
        velocity = np.asarray(velocity, dtype=float)
        gyro_gain = (1.0 if self.gyro_filter_tau == 0 else
                     -np.expm1(-self.dt / self.gyro_filter_tau))
        if self.filtered_gyro is None:
            self.filtered_gyro = gyro.copy() if gyro_gain == 1 else np.zeros(3)
        else:
            self.filtered_gyro += gyro_gain * (gyro - self.filtered_gyro)

        if self.previous_navigation_time is None:
            # The rail position is known; the first noisy GNSS fix is not.
            self.filtered_position = np.array([0., 0., self.elevation])
            self.filtered_velocity = velocity.copy()
            self.navigation_covariance[:, 0, 0] = 1e-4
            self.navigation_covariance[:, 1, 1] = np.maximum(
                self.gnss_velocity_sigma**2, 1e-8)
        else:
            elapsed = self.observation_time - self.previous_navigation_time
            if elapsed <= 0 or not np.isfinite(elapsed):
                elapsed = self.dt
            # Estimate orientation halfway through the observed gyro increment.
            increment = (self.filtered_gyro + self.previous_gyro) * self.dt / 2
            angle = np.linalg.norm(increment)
            if angle > 1e-12:
                half_rotation = np.r_[np.cos(angle / 4),
                                      np.sin(angle / 4) * increment / angle]
                middle_quaternion = _multiply_quaternions(self.quaternion, half_rotation)
            else:
                middle_quaternion = self.quaternion
            # The sensor reports specific force, i.e. inertial acceleration
            # minus gravity.  Restore gravity before propagating navigation.
            world_acceleration = (_rotate_body_to_world(
                middle_quaternion, self.observed_acceleration) + G)
            acceleration_sigma = max(
                self.process_acceleration_sigma,
                self.accelerometer_noise_density / np.sqrt(elapsed),
                self.accelerometer_random_walk_density * np.sqrt(elapsed))
            predicted_position = (self.filtered_position + self.filtered_velocity * elapsed +
                                  0.5 * world_acceleration * elapsed**2)
            self.diagnostics['position_innovation_sum'] += float(
                np.linalg.norm(position - predicted_position))
            self.diagnostics['navigation_updates'] += 1
            (self.filtered_position, self.filtered_velocity,
             self.navigation_covariance) = fuse_navigation_step(
                self.filtered_position, self.filtered_velocity, self.navigation_covariance,
                world_acceleration, position, velocity, elapsed,
                self.gnss_position_sigma, self.gnss_velocity_sigma,
                acceleration_sigma)

        self.previous_navigation_time = self.observation_time
        return (self.filtered_gyro.copy(), self.filtered_position.copy(),
                self.filtered_velocity.copy())
