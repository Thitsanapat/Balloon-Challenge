"""Observation-only, uncertainty-aware ranking of feasible Scenario-4 chains.

This is an experimental planner.  The parent generates and checks feasible
rocket splines; this class only changes the ordering of those candidates.
Forecast errors are measured against *later observations*, never simulator
truth or a saved seed trajectory.  The isotropic Gaussian capture model is a
ranking approximation, not the official collision rule.
"""

from collections import deque
import heapq

import numpy as np
from scipy.special import ndtr

from BalloonPoppingGymEnv.agents.scenario4_wind_profile_agent import (
    Scenario4WindProfileAgent,
)


def ball_capture_probability(distance, radius, sigma):
    """P(||X|| <= radius), X ~ N((distance, 0, 0), sigma**2 I)."""
    distance, sigma = np.broadcast_arrays(
        np.asarray(distance, dtype=float), np.asarray(sigma, dtype=float))
    if radius <= 0 or not np.isfinite(radius) or np.any(distance < 0) or np.any(sigma <= 0):
        raise ValueError("Positive radius, sigma and nonnegative distance required")
    q = radius / sigma
    central = 2 * ndtr(q) - 1 - np.sqrt(2 / np.pi) * q * np.exp(-q * q / 2)
    d = np.maximum(distance, 1e-8)
    lo, hi = (radius - d) / sigma, (radius + d) / sigma
    offset = sigma / (d * np.sqrt(2 * np.pi)) * (
        np.exp(-hi * hi / 2) - np.exp(-lo * lo / 2))
    off_center = ndtr(lo) + ndtr(hi) - 1 + offset
    return np.clip(np.where(distance < 1e-5, central, off_center), 0, 1)


class ForecastErrorHistory:
    """Measure causal 1/2/3-second balloon forecast residuals."""

    def __init__(self, horizons=(1.0, 2.0, 3.0), sample_period=0.5,
                 peers=8, history=128, minimum_samples=12):
        self.horizons = tuple(float(h) for h in horizons)
        self.sample_period = float(sample_period)
        self.peers = int(peers)
        self.minimum_samples = int(minimum_samples)
        if (not self.horizons or min(self.horizons) <= 0 or
                self.sample_period <= 0 or self.peers < 1 or
                history < 1 or self.minimum_samples < 1):
            raise ValueError("Invalid forecast history settings")
        self.pending = []
        self.serial = 0
        self.errors = {h: deque(maxlen=int(history)) for h in self.horizons}
        self.next_sample = -np.inf

    def observe(self, now, states, status, profile, blend, altitude):
        states = np.asarray(states, dtype=float)
        status = np.asarray(status).reshape(-1)
        while self.pending and self.pending[0][0] <= now + 1e-8:
            _, _, index, horizon, prediction = heapq.heappop(self.pending)
            if status[index] == 1 and np.all(np.isfinite(states[index])):
                residual = states[index, :3] - prediction
                self.errors[horizon].append(float(np.dot(residual, residual)))
        if now + 1e-8 < self.next_sample:
            return
        self.next_sample = now + self.sample_period
        valid = np.flatnonzero((status == 1) & np.all(np.isfinite(states), axis=1))
        ids = valid[np.argsort(np.abs(states[valid, 2] - altitude))[:self.peers]]
        for horizon in self.horizons:
            for index in ids:
                index = int(index)
                prediction = states[index, :3] + horizon * states[index, 3:6]
                prediction = prediction.copy()
                prediction[:2] += blend * profile.correction(
                    states, status, index, horizon)
                self.serial += 1
                heapq.heappush(self.pending, (
                    now + horizon, self.serial, index, horizon, prediction))

    def sigma(self, horizon, navigation_sigma):
        """Combined per-axis forecast and navigation spread with a safe prior."""
        horizons = np.asarray(horizon, dtype=float)
        evidence = []
        for h in self.horizons:
            samples = self.errors[h]
            prior = (0.5 + 0.45 * h) ** 2
            empirical = float(np.mean(samples)) / 3 if len(samples) >= self.minimum_samples else prior
            evidence.append(np.sqrt(max(empirical, 0.25)))
        forecast_sigma = np.interp(
            np.maximum(horizons, 0), self.horizons, evidence,
            left=evidence[0], right=evidence[-1])
        forecast_sigma = forecast_sigma * np.sqrt(
            np.maximum(horizons, 0.5) / np.clip(horizons, self.horizons[0], self.horizons[-1]))
        return np.hypot(forecast_sigma, navigation_sigma)

    @property
    def sample_count(self):
        return sum(map(len, self.errors.values()))


def chance_route_score(curves, durations, states, released, route, radius,
                       uncertainty, navigation_sigma, samples=7, legs=2,
                       time_scale=6.0):
    """Expected target and incidental captures along a time-aligned spline.

    The incidental term uses the minimum distance of linearized relative
    motion within each short spline sample interval, then caps each balloon's
    contribution at one.  Only current observed positions/velocities enter.
    """
    durations = np.asarray(durations, dtype=float)
    route = tuple(int(i) for i in route)
    arrivals = np.cumsum(durations)
    planned = ball_capture_probability(
        np.zeros(len(route)), radius, uncertainty.sigma(arrivals, navigation_sigma))
    route_hits = float(np.sum(planned * np.exp(-arrivals / time_scale)))
    ids = np.asarray([int(i) for i in released if int(i) not in route], dtype=int)
    if not len(ids) or not len(curves):
        return route_hits, 0.0
    count = min(len(curves), legs)
    s = np.linspace(0.0, 1.0, samples)
    basis = s[:, None] ** np.arange(6)
    path = np.einsum("sk,nkj->nsj", basis, np.asarray(curves[:count]))
    times = np.r_[0.0, arrivals[:count - 1]][:, None] + durations[:count, None] * s
    targets = states[ids]
    relative = (path[:, :, None, :] - targets[None, None, :, :3]
                - times[:, :, None, None] * targets[None, None, :, 3:6])
    delta = np.diff(relative, axis=1)
    fraction = np.clip(-np.sum(relative[:, :-1] * delta, axis=-1) /
                       np.maximum(np.sum(delta * delta, axis=-1), 1e-12), 0, 1)
    near = relative[:, :-1] + fraction[..., None] * delta
    distance = np.linalg.norm(near, axis=-1)
    near_time = times[:, :-1, None] + fraction * np.diff(times, axis=1)[:, :, None]
    sigma = uncertainty.sigma(near_time, navigation_sigma)
    probability = ball_capture_probability(distance, radius, sigma)
    probability *= np.exp(-near_time / time_scale)
    incidental_hits = float(np.max(probability.reshape(-1, len(ids)), axis=0).sum())
    return route_hits, incidental_hits


class Scenario4ChanceRouteAgent(Scenario4WindProfileAgent):
    """Conservatively reorder baseline-feasible routes by expected captures."""

    def __init__(self, given_parameters, chance_route_weight=1.0,
                 chance_incidental_weight=1.0, chance_navigation_sigma=1.5,
                 chance_samples=7, chance_legs=2, chance_time_scale=6.0,
                 **kwargs):
        super().__init__(given_parameters, **kwargs)
        self.chance_route_weight = float(chance_route_weight)
        self.chance_incidental_weight = float(chance_incidental_weight)
        self.chance_navigation_sigma = float(chance_navigation_sigma)
        self.chance_samples = int(chance_samples)
        self.chance_legs = int(chance_legs)
        self.chance_time_scale = float(chance_time_scale)
        if (not np.all(np.isfinite((self.chance_route_weight,
                self.chance_incidental_weight, self.chance_navigation_sigma,
                self.chance_time_scale))) or self.chance_route_weight < 0 or
                self.chance_incidental_weight < 0 or
                self.chance_navigation_sigma <= 0 or self.chance_samples < 3 or
                self.chance_legs < 1 or self.chance_time_scale <= 0):
            raise ValueError("Invalid chance route settings")
        self.chance_history = ForecastErrorHistory()
        self.diagnostics.update(chance_ranked_nodes=0,
                                chance_forecast_samples=0,
                                chance_selected_expectations=[])

    def get_action(self, observation):
        now = float(observation["simulation_time"])
        altitude = (float(self.filtered_position[2])
                    if self.filtered_position is not None else self.elevation)
        self.chance_history.observe(
            now, observation["balloon_states"], observation["balloon_status"],
            self.wind_profile, self.profile_blend, altitude)
        action = super().get_action(observation)
        self.diagnostics["chance_forecast_samples"] = self.chance_history.sample_count
        return action

    def _node_priority(self, node, states, released, now):
        base = super()._node_priority(node, states, released, now)
        if (not node[0] or (self.chance_route_weight == 0 and
                            self.chance_incidental_weight == 0)):
            return base
        planned, incidental = chance_route_score(
            node[2], node[1], states, released, node[0], self.radius,
            self.chance_history, self.chance_navigation_sigma,
            self.chance_samples, self.chance_legs, self.chance_time_scale)
        self.diagnostics["chance_ranked_nodes"] += 1
        # At a fixed beam depth, a perfectly certain route has zero penalty.
        return (base + self.chance_route_weight * (len(node[0]) - planned)
                - self.chance_incidental_weight * incidental)

    def _make_plan(self, observation, position, velocity, now):
        events = len(self.route_events)
        super()._make_plan(observation, position, velocity, now)
        if len(self.route_events) == events or self.plan is None:
            return
        states = np.asarray(observation["balloon_states"], dtype=float)
        released = np.flatnonzero(
            np.asarray(observation["balloon_status"]).reshape(-1) == 1)
        planned, incidental = chance_route_score(
            self.plan[None, :, :], np.array([self.plan_duration]),
            states, released, self.route, self.radius,
            self.chance_history, self.chance_navigation_sigma,
            self.chance_samples, 1, self.chance_time_scale)
        self.diagnostics["chance_selected_expectations"].append(
            [float(now), float(planned), float(incidental)])
