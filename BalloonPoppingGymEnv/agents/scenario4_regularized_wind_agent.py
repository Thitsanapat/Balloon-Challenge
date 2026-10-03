"""Experimental altitude-drift forecast using only live balloon observations.

The fitted profile is a robust, ridge-regularized estimate of the *observed*
horizontal balloon velocity at each altitude. It is not an atmospheric-field
lookup. Predictions remain anchored to the target's own measured velocity.
"""

import numpy as np

from BalloonPoppingGymEnv.agents.scenario4_wind_profile_agent import (
    Scenario4WindProfileAgent,
)


class RegularizedObservedWindProfile:
    """Locally estimate altitude-dependent drift with smooth edge tapering."""

    def __init__(self, width=20.0, min_neighbors=5, min_spread=5.0,
                 ridge_spread=8.0, max_extrapolation=6.0,
                 max_velocity_difference=4.0, max_position_correction=12.0,
                 integration_steps=9):
        self.width = float(width)
        self.min_neighbors = int(min_neighbors)
        self.min_spread = float(min_spread)
        self.ridge_spread = float(ridge_spread)
        self.max_extrapolation = float(max_extrapolation)
        self.max_velocity_difference = float(max_velocity_difference)
        self.max_position_correction = float(max_position_correction)
        self.integration_steps = int(integration_steps)
        finite = (self.width, self.min_spread, self.ridge_spread,
                  self.max_extrapolation, self.max_velocity_difference,
                  self.max_position_correction)
        if (not np.all(np.isfinite(finite)) or self.width <= 0 or
                self.min_neighbors < 3 or self.min_spread <= 0 or
                self.ridge_spread < 0 or self.max_extrapolation < 0 or
                self.max_velocity_difference <= 0 or
                self.max_position_correction <= 0 or self.integration_steps < 3):
            raise ValueError("Invalid regularized wind profile settings")

    def _local_velocity(self, peers, altitude):
        distance = peers[:, 2] - altitude
        local = np.abs(distance) <= 2.5 * self.width
        if np.count_nonzero(local) < self.min_neighbors:
            return None
        z = peers[local, 2]
        if (altitude < np.min(z) - self.max_extrapolation or
                altitude > np.max(z) + self.max_extrapolation or
                np.std(z) < self.min_spread):
            return None
        x = distance[local]
        velocity = peers[local, 3:5]
        weight = np.exp(-0.5 * (x / self.width) ** 2)

        def fit(w):
            total = float(np.sum(w))
            mean_x = float(np.dot(w, x) / total)
            mean_v = np.einsum("n,nj->j", w, velocity) / total
            centered = x - mean_x
            slope = (np.einsum("n,n,nj->j", w, centered, velocity) /
                     (np.dot(w, centered ** 2) + total * self.ridge_spread ** 2))
            return mean_v - mean_x * slope, slope

        # Start from a high-breakdown fit. Initial ordinary least squares can
        # tilt toward one unusually fast balloon so far that Huber iterations
        # reject good peers instead of the outlier.
        separation = x[None, :] - x[:, None]
        pairs = np.triu(np.abs(separation) >= self.min_spread, k=1)
        if np.count_nonzero(pairs) < self.min_neighbors:
            return None
        velocity_difference = velocity[None, :, :] - velocity[:, None, :]
        slope = np.median(
            velocity_difference[pairs] / separation[pairs, None], axis=0,
        )
        intercept = np.median(velocity - x[:, None] * slope, axis=0)
        for _ in range(2):
            residual = np.linalg.norm(
                velocity - intercept - x[:, None] * slope, axis=1,
            )
            threshold = max(0.75, 2.5 * float(np.median(residual)))
            robust = np.minimum(1.0, threshold / np.maximum(residual, 1e-12))
            intercept, slope = fit(weight * robust)
        return intercept

    def correction(self, states, status, index, horizon):
        """Return a bounded horizontal displacement beyond constant velocity."""
        states = np.asarray(states, dtype=float)
        status = np.asarray(status).reshape(-1)
        index = int(index)
        horizon = float(horizon)
        if (states.ndim != 2 or states.shape[1] != 6 or
                status.shape != (len(states),) or not 0 <= index < len(states)
                or not np.isfinite(horizon) or horizon <= 0):
            return np.zeros(2)
        target = states[index]
        if status[index] != 1 or not np.all(np.isfinite(target)):
            return np.zeros(2)
        mask = (status == 1) & np.all(np.isfinite(states), axis=1)
        mask[index] = False
        peers = states[mask]
        if len(peers) < self.min_neighbors:
            return np.zeros(2)
        z0 = float(target[2])
        initial = self._local_velocity(peers, z0)
        if initial is None:
            return np.zeros(2)

        times = np.linspace(0.0, horizon, self.integration_steps)
        changes = []
        last_change = np.zeros(2)
        last_supported_altitude = z0
        for elapsed in times:
            altitude = z0 + target[5] * elapsed
            estimated = self._local_velocity(peers, altitude)
            if estimated is None:
                # Do not discard the part of the path with observed support.
                # Unseen altitude progressively reverts to constant velocity.
                change = last_change * np.exp(
                    -abs(altitude - last_supported_altitude) / self.width,
                )
            else:
                change = estimated - initial
                last_supported_altitude = altitude
            magnitude = float(np.linalg.norm(change))
            if magnitude > self.max_velocity_difference:
                change = change * self.max_velocity_difference / magnitude
            if estimated is not None:
                last_change = change
            changes.append(change)
        offset = np.trapezoid(np.asarray(changes), times, axis=0)
        length = float(np.linalg.norm(offset))
        if length > self.max_position_correction:
            offset *= self.max_position_correction / length
        return offset


class Scenario4RegularizedWindAgent(Scenario4WindProfileAgent):
    """Use the regularized forecast in the established route planner."""

    def __init__(self, given_parameters, profile_ridge_spread=8.0, **kwargs):
        super().__init__(given_parameters, **kwargs)
        old = self.wind_profile
        self.wind_profile = RegularizedObservedWindProfile(
            width=old.width,
            min_neighbors=old.min_neighbors,
            min_spread=old.min_spread,
            ridge_spread=profile_ridge_spread,
            max_velocity_difference=old.max_velocity_difference,
            max_position_correction=old.max_position_correction,
            integration_steps=old.integration_steps,
        )
