"""Broaden beam candidates using time-aligned public-model intercept proxies.

This only orders observed, released targets; the ordinary joint spline search
and dense actuator checks still decide feasibility. No seed or saved route is
available to the policy. Constant acceleration is a screening proxy, not a
claim that the actual rocket can instantaneously attain that acceleration.
"""

import numpy as np

from BalloonPoppingGymEnv.agents.scenario4_wind_profile_agent import Scenario4WindProfileAgent
from BalloonPoppingGymEnv.agents.submission_sqp_v2 import G


def intercept_proxy_scores(states, position, velocity, elapsed, horizons,
                           limits, disturbance, max_tilt):
    """Rank time-synchronized intercept effort, penalizing envelope excess."""
    states = np.asarray(states, dtype=float)
    times = np.asarray(horizons, dtype=float)
    limits = np.asarray(limits, dtype=float)
    if (states.ndim != 2 or states.shape[1] != 6 or times.ndim != 1
            or not len(times) or limits.shape != times.shape
            or not np.all(np.isfinite(times)) or np.any(times <= 0)
            or not np.all(np.isfinite(limits)) or np.any(limits <= 0)):
        raise ValueError("Finite positive horizons and thrust limits required")
    target = states[:, None, :3] + states[:, None, 3:] * (elapsed + times)[None, :, None]
    displacement = target - np.asarray(position) - np.asarray(velocity) * times[:, None]
    force = 2 * displacement / times[None, :, None]**2 - G - np.asarray(disturbance)
    magnitude = np.linalg.norm(force, axis=2)
    effort = magnitude / limits[None, :]
    vertical = force[:, :, 2] / np.maximum(magnitude, 1e-9)
    excess = np.maximum(effort - 1, 0) + np.maximum(np.cos(max_tilt) - vertical, 0)
    score = times[None, :] + 2 * effort**2 + 40 * excess
    score[~np.isfinite(score)] = np.inf
    return np.min(score, axis=1)


class Scenario4InterceptCandidatesAgent(Scenario4WindProfileAgent):
    def __init__(self, given_parameters, candidate_keep_fraction=0.5,
                 candidate_time_samples=9, **kwargs):
        super().__init__(given_parameters, **kwargs)
        self.candidate_keep_fraction = float(candidate_keep_fraction)
        self.candidate_time_samples = int(candidate_time_samples)
        if (not np.isfinite(self.candidate_keep_fraction)
                or not 0 <= self.candidate_keep_fraction <= 1
                or self.candidate_time_samples < 2):
            raise ValueError("Invalid candidate screening settings")
        self.diagnostics.update(intercept_candidate_calls=0, intercept_candidate_changes=0)

    def _candidate_ids(self, states, released, route, elapsed, position, velocity, now):
        baseline = super()._candidate_ids(states, released, route, elapsed, position, velocity, now)
        if self.candidate_keep_fraction == 1:
            return baseline
        remaining = self.launch_time + self.burn_time - now - elapsed - 1e-4
        upper = min(self.leg_horizon, remaining)
        lower = 1.0 if route else 2.0
        if upper < lower:
            return baseline
        ids = [int(i) for i in released if i not in route
               and self.failed_until.get(int(i), 0) <= now
               and np.all(np.isfinite(states[i]))]
        if not ids:
            return []
        horizons = np.linspace(lower, upper, self.candidate_time_samples)
        limits = np.array([self.available_acceleration(now + elapsed + t / 2)
                           for t in horizons]) - self.reserve
        scores = intercept_proxy_scores(states[ids], position, velocity, elapsed,
                                        horizons, np.maximum(limits, 0.1),
                                        self.disturbance, self.max_tilt)
        keep = int(np.ceil(self.branch_targets * self.candidate_keep_fraction))
        selected = baseline[:keep]
        for offset in np.argsort(scores, kind="stable"):
            index = ids[int(offset)]
            if index not in selected:
                selected.append(index)
            if len(selected) >= self.branch_targets:
                break
        self.diagnostics['intercept_candidate_calls'] += 1
        self.diagnostics['intercept_candidate_changes'] += int(set(selected) != set(baseline))
        return selected
