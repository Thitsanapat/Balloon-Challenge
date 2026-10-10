"""Observation-triggered launch timing for Scenario 4.

This variant keeps the validated wind-profile route planner intact.  Between
an earliest and a default launch time it may launch early if a *released*,
time-aligned balloon group has a plausible first intercept under a simple
published-thrust bound.  Otherwise it launches at the default time.  The
seed, hidden wind, unreleased balloons, and future simulator state are never
read.  The group calculation only triggers a full ordinary launch search; it
does not bypass its trajectory feasibility checks.
"""

import numpy as np

from BalloonPoppingGymEnv.agents.scenario4_wind_profile_agent import (
    Scenario4WindProfileAgent,
)
from BalloonPoppingGymEnv.agents.submission_sqp_v2 import G


def observed_launch_group(states, released, position, available,
                          max_intercept_time=10.0, group_radius=8.0,
                          step=0.5):
    """Return the largest reachable group and its first-hit time proxy.

    A constant-acceleration bound from the rail is only a cheap launch
    readiness test. The actual chain planner subsequently validates thrust,
    gimbal and ground clearance along a full trajectory.
    """
    states = np.asarray(states, dtype=float)
    ids = np.asarray(released, dtype=int)
    position = np.asarray(position, dtype=float)
    if (states.ndim != 2 or states.shape[1] != 6 or position.shape != (3,)
            or not np.all(np.isfinite(position)) or
            not np.isfinite(available) or available <= 0 or
            not np.isfinite(max_intercept_time) or max_intercept_time < 2 or
            not np.isfinite(group_radius) or group_radius <= 0 or
            not np.isfinite(step) or step <= 0):
        raise ValueError("Invalid launch-group inputs")
    if not len(ids):
        return 0, float("inf")
    observed = states[ids]
    if not np.all(np.isfinite(observed)):
        raise ValueError("Released balloon states must be finite")
    best_count, best_time = 0, float("inf")
    for duration in np.arange(2.0, max_intercept_time + 1e-9, step):
        projected = observed[:, :3] + duration * observed[:, 3:6]
        acceleration = 2 * (projected - position) / duration**2
        force = acceleration - G
        norm = np.linalg.norm(force, axis=1)
        reachable = ((norm <= 0.98 * available) &
                     (acceleration[:, 2] > 0.1) &
                     (force[:, 2] >= norm * np.cos(np.radians(35.0))))
        if not np.any(reachable):
            continue
        centers = projected[reachable]
        counts = np.count_nonzero(
            np.linalg.norm(centers[:, None, :] -
                           projected[None, :, :], axis=2) <= group_radius,
            axis=1)
        peak = int(np.max(counts))
        if peak > best_count or (peak == best_count and duration < best_time):
            best_count, best_time = peak, float(duration)
    return best_count, best_time


class Scenario4AdaptiveLaunchAgent(Scenario4WindProfileAgent):
    """Launch early only for a plausible observed group, else use baseline."""

    def __init__(self, given_parameters, earliest_launch=40.0,
                 default_launch=42.0, launch_check_interval=0.5,
                 launch_min_group=4, launch_group_radius=8.0,
                 launch_max_intercept_time=10.0, **kwargs):
        if "launch_time" in kwargs:
            raise ValueError("Use default_launch, not launch_time")
        self.earliest_launch = float(earliest_launch)
        self.default_launch = float(default_launch)
        self.launch_check_interval = float(launch_check_interval)
        self.launch_min_group = int(launch_min_group)
        self.launch_group_radius = float(launch_group_radius)
        self.launch_max_intercept_time = float(launch_max_intercept_time)
        values = (self.earliest_launch, self.default_launch,
                  self.launch_check_interval, self.launch_group_radius,
                  self.launch_max_intercept_time)
        if (not np.all(np.isfinite(values)) or self.earliest_launch < 0 or
                self.default_launch < self.earliest_launch or
                self.launch_check_interval <= 0 or self.launch_min_group < 2 or
                self.launch_group_radius <= 0 or
                self.launch_max_intercept_time < 2):
            raise ValueError("Invalid adaptive-launch settings")
        super().__init__(given_parameters, launch_time=self.default_launch,
                         **kwargs)
        self.next_launch_check = self.earliest_launch
        self.diagnostics.update(launch_readiness_checks=0,
                                adaptive_launch_reason=None,
                                adaptive_launch_group=0,
                                launch_group_history=[])

    def get_action(self, observation):
        now = float(observation["simulation_time"])
        if not self.launched and now >= self.earliest_launch:
            if (now >= self.next_launch_check and
                    now < self.default_launch):
                states = np.asarray(observation["balloon_states"], dtype=float)
                released = np.flatnonzero(
                    np.asarray(observation["balloon_status"]).reshape(-1) == 1)
                # The vehicle sheds mass over the first leg, so initial
                # thrust-to-mass alone is too pessimistic for a ~9-s hit.
                # This public-parameter mid-burn proxy is only a trigger;
                # the normal full-trajectory checks still choose the route.
                mid_burn_age = min(0.5 * self.launch_max_intercept_time,
                                   self.burn_time)
                mid_burn_available = self.thrust / max(
                    self.dry_mass,
                    self.initial_mass - self.mass_flow * mid_burn_age)
                group, intercept_time = observed_launch_group(
                    states, released, np.array([0., 0., self.elevation]),
                    mid_burn_available,
                    max_intercept_time=self.launch_max_intercept_time,
                    group_radius=self.launch_group_radius)
                self.diagnostics["launch_readiness_checks"] += 1
                self.diagnostics["launch_group_history"].append(
                    [float(now), int(group),
                     float(intercept_time) if np.isfinite(intercept_time)
                     else None])
                self.next_launch_check = now + self.launch_check_interval
                if group >= self.launch_min_group:
                    self.launch_time = now
                    self.diagnostics["adaptive_launch_reason"] = "observed_group"
                    self.diagnostics["adaptive_launch_group"] = group
            if (now >= self.default_launch and
                    self.diagnostics["adaptive_launch_reason"] is None):
                self.diagnostics["adaptive_launch_reason"] = "default_time"
        return super().get_action(observation)
