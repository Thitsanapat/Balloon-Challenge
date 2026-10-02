"""Short-range target prediction from the observed balloon wind shear.

This experimental policy uses only the current observation and the public
``given_parameters``.  It does not inspect the simulator's atmospheric field,
random seed, or prerecorded trajectories.  Nearby *released* balloons provide
an estimate of how horizontal drift changes with altitude.  The estimate is
anchored to each target's own measured velocity, so differences in individual
balloon dynamics are not treated as a replacement for that measurement.
"""

import numpy as np

from BalloonPoppingGymEnv.agents.time_allocation_agent import TimeAllocationAgent


class ObservedWindProfile:
    """Robust local horizontal drift as a function of observed altitude."""

    def __init__(self, width=20.0, min_neighbors=5, min_spread=5.0,
                 max_extrapolation=3.0, max_velocity_difference=4.0,
                 max_position_correction=12.0, integration_steps=9):
        self.width = float(width)
        self.min_neighbors = int(min_neighbors)
        self.min_spread = float(min_spread)
        self.max_extrapolation = float(max_extrapolation)
        self.max_velocity_difference = float(max_velocity_difference)
        self.max_position_correction = float(max_position_correction)
        self.integration_steps = int(integration_steps)
        scalars = (self.width, self.min_spread, self.max_extrapolation,
                   self.max_velocity_difference, self.max_position_correction)
        if (not np.all(np.isfinite(scalars)) or self.width <= 0 or
                self.min_neighbors < 3 or self.min_spread <= 0 or
                self.max_extrapolation < 0 or self.max_velocity_difference <= 0 or
                self.max_position_correction <= 0 or self.integration_steps < 3):
            raise ValueError("Invalid observed wind profile settings")

    def _local_velocity(self, peers, altitude):
        """Median pairwise slope and intercept: one anomalous peer is harmless."""
        local = peers[np.abs(peers[:, 2] - altitude) <= 2 * self.width]
        if len(local) < self.min_neighbors:
            return None
        # Limit the fit to the nearest observations so a distant shear reversal
        # cannot overwrite the local slope. The fit need not know gust nodes.
        nearest = np.argsort(np.abs(local[:, 2] - altitude))
        local = local[nearest[:max(self.min_neighbors, 12)]]
        z = local[:, 2]
        if (altitude < np.min(z) - self.max_extrapolation or
                altitude > np.max(z) + self.max_extrapolation or
                np.std(z) < self.min_spread):
            return None
        dz = z[None, :] - z[:, None]
        pairs = np.triu(np.abs(dz) >= self.min_spread, k=1)
        if np.count_nonzero(pairs) < self.min_neighbors:
            return None
        dv = local[None, :, 3:5] - local[:, None, 3:5]
        slope = np.median(dv[pairs] / dz[pairs, None], axis=0)
        intercept = np.median(local[:, 3:5] -
                              (z - altitude)[:, None] * slope, axis=0)
        return intercept

    def correction(self, states, status, index, horizon):
        """Return a bounded horizontal offset at ``horizon`` seconds.

        The target's measured current velocity is the anchor. Integrating the
        *change* in profile velocity as it rises predicts displacement beyond
        the ordinary constant-velocity projection. Sparse or unsupported
        altitude regions fall back to that ordinary projection.
        """
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
        z0 = float(target[2])
        mask = (status == 1) & np.all(np.isfinite(states), axis=1)
        mask[index] = False
        peers = states[mask]
        if len(peers) < self.min_neighbors:
            return np.zeros(2)
        initial_velocity = self._local_velocity(peers, z0)
        if initial_velocity is None:
            return np.zeros(2)
        times = np.linspace(0.0, horizon, self.integration_steps)
        changes = []
        for elapsed in times:
            estimated = self._local_velocity(peers, z0 + target[5] * elapsed)
            if estimated is None:
                return np.zeros(2)
            change = estimated - initial_velocity
            magnitude = float(np.linalg.norm(change))
            if magnitude > self.max_velocity_difference:
                change *= self.max_velocity_difference / magnitude
            changes.append(change)
        correction = np.trapezoid(np.asarray(changes), times, axis=0)
        length = float(np.linalg.norm(correction))
        if length > self.max_position_correction:
            correction *= self.max_position_correction / length
        return correction


class Scenario4WindProfileAgent(TimeAllocationAgent):
    """Feed a bounded altitude-aware drift estimate to the existing planner."""

    def __init__(self, given_parameters, profile_horizon=8.0,
                 profile_max_range=150.0, profile_nominal_speed=14.0,
                 profile_blend=0.75, profile_width=20.0,
                 profile_min_neighbors=5, profile_min_spread=5.0,
                 profile_max_extrapolation=3.0,
                 profile_max_velocity_difference=4.0,
                 profile_max_position_correction=12.0, **kwargs):
        super().__init__(given_parameters, **kwargs)
        self.profile_horizon = float(profile_horizon)
        self.profile_max_range = float(profile_max_range)
        self.profile_nominal_speed = float(profile_nominal_speed)
        self.profile_blend = float(profile_blend)
        settings = (self.profile_horizon, self.profile_max_range,
                    self.profile_nominal_speed, self.profile_blend)
        if (not np.all(np.isfinite(settings)) or
                min(settings[:3]) <= 0 or not 0 <= self.profile_blend <= 1):
            raise ValueError("Invalid wind-profile forecast settings")
        self.wind_profile = ObservedWindProfile(
            width=profile_width, min_neighbors=profile_min_neighbors,
            min_spread=profile_min_spread,
            max_extrapolation=profile_max_extrapolation,
            max_velocity_difference=profile_max_velocity_difference,
            max_position_correction=profile_max_position_correction,
        )
        self.diagnostics.update(profile_plans=0, profile_corrected_targets=0)

    def _make_plan(self, observation, position, velocity, now):
        states = np.asarray(observation["balloon_states"], dtype=float)
        status = np.asarray(observation["balloon_status"]).reshape(-1)
        position = np.asarray(position, dtype=float)
        velocity = np.asarray(velocity, dtype=float)
        adjusted = None
        corrected = 0
        active = self.route[0] if self.route else None
        for index in np.flatnonzero(status == 1):
            relative = states[index, :3] - position
            distance = float(np.linalg.norm(relative))
            if not np.isfinite(distance) or distance > self.profile_max_range:
                continue
            closing = float(np.dot(relative, velocity - states[index, 3:6]) /
                            max(distance, 1e-9))
            horizon = distance / max(self.profile_nominal_speed, closing)
            if index == active and self.deadlines:
                horizon = float(self.deadlines[0] - now)
            if not 0 < horizon <= self.profile_horizon:
                continue
            offset = self.wind_profile.correction(states, status, index, horizon)
            if not np.any(offset):
                continue
            if adjusted is None:
                adjusted = states.copy()
            # Changing the effective drift (rather than the current position)
            # leaves the observed target location exact at t=0 and matches the
            # shear-corrected prediction at the expected interception time.
            adjusted[index, 3:5] += self.profile_blend * offset / horizon
            corrected += 1
        if corrected:
            observation = dict(observation, balloon_states=adjusted)
            self.diagnostics["profile_plans"] += 1
            self.diagnostics["profile_corrected_targets"] += corrected
        return super()._make_plan(observation, position, velocity, now)
