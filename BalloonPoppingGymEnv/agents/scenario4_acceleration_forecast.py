"""Observation-only short-horizon balloon acceleration forecast.

The wind-profile planner already accounts for vertical shear.  This predictor
tracks *measured* target acceleration and adds only the part not explained by
the profile's instantaneous shear.  Its correction decays with horizon, is
bounded, and has no access to simulator wind fields or future trajectories.
"""

import numpy as np


def decaying_acceleration_displacement(acceleration, horizon, decay_seconds):
    """Integrate an exponentially decaying acceleration over ``horizon``."""
    acceleration = np.asarray(acceleration, dtype=float)
    if (acceleration.shape != (2,) or not np.all(np.isfinite(acceleration)) or
            not np.isfinite(horizon) or horizon < 0 or
            not np.isfinite(decay_seconds) or decay_seconds <= 0):
        raise ValueError("Invalid acceleration forecast inputs")
    t = float(horizon)
    decay = float(decay_seconds)
    area = decay * (t + decay * np.expm1(-t/decay))
    return acceleration * area


class ObservedAccelerationForecast:
    """Causal per-balloon acceleration estimate from released observations."""

    def __init__(self, count, filter_seconds=0.25, decay_seconds=1.2,
                 strength=0.5, max_acceleration=3.0,
                 max_residual_acceleration=1.5,
                 max_position_correction=4.0, profile_probe_seconds=0.5):
        if not isinstance(count, int) or count <= 0:
            raise ValueError("Positive balloon count required")
        values = (filter_seconds, decay_seconds, strength, max_acceleration,
                  max_residual_acceleration, max_position_correction,
                  profile_probe_seconds)
        if (not np.all(np.isfinite(values)) or filter_seconds <= 0 or
                decay_seconds <= 0 or not 0 <= strength <= 1 or
                min(max_acceleration, max_residual_acceleration,
                    max_position_correction, profile_probe_seconds) <= 0):
            raise ValueError("Invalid acceleration forecast settings")
        self.count = count
        self.filter_seconds = float(filter_seconds)
        self.decay_seconds = float(decay_seconds)
        self.strength = float(strength)
        self.max_acceleration = float(max_acceleration)
        self.max_residual_acceleration = float(max_residual_acceleration)
        self.max_position_correction = float(max_position_correction)
        self.profile_probe_seconds = float(profile_probe_seconds)
        self.previous_time = None
        self.previous_velocity = np.zeros((count, 2))
        self.previous_valid = np.zeros(count, dtype=bool)
        self.acceleration = np.zeros((count, 2))

    def observe(self, time, states, status):
        states = np.asarray(states, dtype=float)
        status = np.asarray(status).reshape(-1)
        time = float(time)
        if (states.shape != (self.count, 6) or status.shape != (self.count,) or
                not np.isfinite(time)):
            raise ValueError("Invalid balloon observation")
        valid = (status == 1) & np.all(np.isfinite(states), axis=1)
        if self.previous_time is not None:
            dt = time - self.previous_time
            if dt <= 0:
                raise ValueError("Observation time must increase")
            mask = valid & self.previous_valid
            measured = np.zeros_like(self.acceleration)
            measured[mask] = (states[mask, 3:5] -
                              self.previous_velocity[mask]) / dt
            sizes = np.linalg.norm(measured[mask], axis=1)
            if len(sizes):
                measured[mask] *= (np.minimum(1.0,
                    self.max_acceleration / np.maximum(sizes, 1e-12)))[:, None]
            gain = -np.expm1(-dt/self.filter_seconds)
            self.acceleration[mask] += gain * (
                measured[mask] - self.acceleration[mask])
            self.acceleration[~mask] = 0.0
        self.previous_time = time
        self.previous_velocity = states[:, 3:5].copy()
        self.previous_valid = valid

    def correction(self, states, status, index, horizon, wind_profile):
        """Residual offset beyond the caller's wind-profile forecast."""
        states = np.asarray(states, dtype=float)
        status = np.asarray(status).reshape(-1)
        if (states.shape != (self.count, 6) or status.shape != (self.count,) or
                not 0 <= index < self.count or status[index] != 1 or
                not np.isfinite(horizon) or horizon <= 0 or
                not np.all(np.isfinite(states[index]))):
            return np.zeros(2)
        probe = self.profile_probe_seconds
        profile_accel = 2 * wind_profile.correction(
            states, status, index, probe) / probe**2
        residual = self.acceleration[index] - profile_accel
        size = float(np.linalg.norm(residual))
        if size > self.max_residual_acceleration:
            residual *= self.max_residual_acceleration / size
        offset = self.strength * decaying_acceleration_displacement(
            residual, horizon, self.decay_seconds)
        length = float(np.linalg.norm(offset))
        if length > self.max_position_correction:
            offset *= self.max_position_correction / length
        return offset
