"""Observation-only navigation/control ablations atop joint route refresh.

The optional navigation change integrates adjacent GNSS velocity samples,
avoiding reuse of the delayed low-pass velocity in position propagation.
The optional controller learns a bounded scalar angular response per axis
from past gyro samples and its own predicted actuator outputs. This is a
small adaptive-gain experiment, not a full INDI controller or 6-DOF model.
"""

import numpy as np

from BalloonPoppingGymEnv.agents.scenario4_pure_joint_refresh_agent import (
    Scenario4PureJointRefreshAgent,
)


class AngularResponseEstimator:
    """Exponentially weighted regression with prior, excitation and slew gates."""

    def __init__(self, dt, gyro_tau, limit=0.15, slew=0.03):
        values = (dt, gyro_tau, limit, slew)
        if (not np.all(np.isfinite(values)) or dt <= 0 or gyro_tau < 0 or
                not 0 <= limit < 1 or slew <= 0):
            raise ValueError('Invalid angular response settings')
        self.dt, self.limit, self.slew = dt, limit, slew
        self.gyro_gain = 1.0 if gyro_tau == 0 else -np.expm1(-dt / gyro_tau)
        self.lowpass_gain = -np.expm1(-dt / 0.10)
        self.decay = np.exp(-dt / 3.0)
        self.nominal_gyro_filtered = np.zeros(2)
        self.nominal = np.zeros(2)
        self.observed = np.zeros(2)
        self.xx = np.zeros(2)
        self.xy = np.zeros(2)
        self.gain = np.ones(2)
        self.excited_seconds = np.zeros(2)
        self.samples = 0
        self.nominal_error_square = 0.0
        self.adapted_error_square = 0.0

    def update(self, nominal, observed, usable=True):
        nominal, observed = np.asarray(nominal), np.asarray(observed)
        if (nominal.shape != (2,) or observed.shape != (2,) or
                not np.all(np.isfinite(np.r_[nominal, observed]))):
            return self.gain.copy()
        # Observed is the derivative of filtered gyro. Match that same lag on
        # the nominal response before applying a common derivative-noise filter.
        self.nominal_gyro_filtered += self.gyro_gain * (
            nominal - self.nominal_gyro_filtered)
        self.nominal += self.lowpass_gain * (self.nominal_gyro_filtered - self.nominal)
        self.observed += self.lowpass_gain * (observed - self.observed)
        self.xx *= self.decay
        self.xy *= self.decay
        excited = usable & (np.abs(self.nominal) > 0.08)
        plausible = np.abs(self.observed) < 4.0
        mask = excited & plausible
        if np.any(mask):
            # Score prediction with the PRE-UPDATE coefficients, not a fitted
            # residual from the same sample. Correlated samples are not a holdout.
            self.nominal_error_square += float(np.sum(
                (self.observed[mask] - self.nominal[mask]) ** 2))
            self.adapted_error_square += float(np.sum(
                (self.observed[mask] - self.gain[mask] * self.nominal[mask]) ** 2))
            self.samples += int(np.count_nonzero(mask))
            self.excited_seconds[mask] += self.dt
            self.xx[mask] += self.dt * self.nominal[mask] ** 2
            self.xy[mask] += self.dt * self.nominal[mask] * self.observed[mask]
        proposed = np.clip((self.xy + 0.02) / (self.xx + 0.02),
                           1 - self.limit, 1 + self.limit)
        ready = mask & (self.excited_seconds >= 1.0) & (self.xx >= 0.015)
        self.gain[ready] += np.clip(proposed[ready] - self.gain[ready],
                                   -self.slew * self.dt, self.slew * self.dt)
        return self.gain.copy()


class Scenario4SensorCoupledAgent(Scenario4PureJointRefreshAgent):
    def __init__(self, given_parameters, navigation_trapezoid=False,
                 adaptive_angular_response=False, response_limit=0.15,
                 response_slew=0.03, **kwargs):
        super().__init__(given_parameters, **kwargs)
        self.navigation_trapezoid = bool(navigation_trapezoid)
        self.adaptive_angular_response = bool(adaptive_angular_response)
        self.angular_response = AngularResponseEstimator(
            self.dt, self.gyro_filter_tau, response_limit, response_slew)
        self._raw_velocity_previous = None
        self._response_time = 0.0
        self.diagnostics.update(sensor_position_updates=0,
                                adaptive_control_steps=0)

    def get_action(self, observation):
        self._response_time = float(observation['simulation_time'])
        return super().get_action(observation)

    def _filter_sensors(self, gyro, position, velocity):
        old_position = (None if self.filtered_position is None else
                        self.filtered_position.copy())
        old_gyro = None if self.filtered_gyro is None else self.filtered_gyro.copy()
        old_filtered_velocity = (None if self.filtered_velocity is None else
                                 self.filtered_velocity.copy())
        result = super()._filter_sensors(gyro, position, velocity)
        if (self.navigation_trapezoid and old_position is not None and
                self._raw_velocity_previous is not None):
            pos_gain = (1.0 if self.position_filter_tau == 0 else
                        -np.expm1(-self.dt / self.position_filter_tau))
            predicted = old_position + 0.5 * self.dt * (
                self._raw_velocity_previous + np.asarray(velocity))
            self.filtered_position = predicted + pos_gain * (position - predicted)
            # Replace the diagnostic contribution already made by the parent.
            old_prediction = old_position + self.dt * old_filtered_velocity
            self.diagnostics['position_innovation_sum'] += float(
                np.linalg.norm(position - predicted) -
                np.linalg.norm(position - old_prediction))
            self.diagnostics['sensor_position_updates'] += 1
            result = (result[0], self.filtered_position.copy(), result[2])
        self._raw_velocity_previous = np.asarray(velocity, dtype=float).copy()
        if self.adaptive_angular_response and old_gyro is not None:
            nominal = (self.thrust * self.previous_throttle * self.lever *
                       np.sin(np.radians(self.previous_tvc)) / self.inertia[:2])
            observed = (result[0][:2] - old_gyro[:2]) / self.dt
            elapsed = self._response_time - self.launch_time
            usable = (1.0 < elapsed < self.burn_time and
                      self.previous_throttle > 0.15 and
                      np.max(np.abs(self.previous_tvc)) <
                      0.9 * self.control['max_gimbal_angle'])
            self.angular_response.update(nominal, observed, usable)
            self.diagnostics.update(
                angular_response_gain=self.angular_response.gain.tolist(),
                angular_response_samples=self.angular_response.samples,
                angular_nominal_error_square=self.angular_response.nominal_error_square,
                angular_adapted_error_square=self.angular_response.adapted_error_square)
        return result

    def _attitude_action(self, axis, jerk, magnitude, gyro, available):
        if not self.adaptive_angular_response:
            return super()._attitude_action(axis, jerk, magnitude, gyro, available)
        original_inertia = self.inertia
        # Confine compensation to the attitude controller. Route constraints
        # and the response estimator continue to use the nominal public model.
        self.inertia = original_inertia.copy()
        self.inertia[:2] /= self.angular_response.gain
        try:
            action = super()._attitude_action(axis, jerk, magnitude, gyro, available)
        finally:
            self.inertia = original_inertia
        self.diagnostics['adaptive_control_steps'] += int(
            np.max(np.abs(self.angular_response.gain - 1)) > 1e-3)
        return action
