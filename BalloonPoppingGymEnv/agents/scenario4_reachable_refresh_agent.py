"""Experimental early refresh of an actuator-limited committed intercept.

Only the current observation, the agent's own filtered state, and public
actuator/motor parameters enter this calculation.  The preview is a *screen*,
not a 6-DOF simulation or a guarantee of capture.  The selected qualification
agent is intentionally not changed by this experiment.
"""

import numpy as np

from BalloonPoppingGymEnv.agents.chain_beam_agent import free_chain
from BalloonPoppingGymEnv.agents.scenario4_guarded_innovation_agent import (
    Scenario4GuardedInnovationAgent,
)
from BalloonPoppingGymEnv.agents.submission_sqp_v2 import (
    G, _rotate_body_to_world, sample_curve,
)


def preview_actuator_shortfall(curve, duration, now, disturbance, body_axis,
                               throttle, acceleration_limit, axis_rate_limit,
                               throttle_rate_limit, horizon=1.2,
                               response_allowance=0.2):
    """Estimate lost displacement when the first part of a spline asks too much.

    An intentionally generous reachable cone and throttle envelope start at
    the *observed* body axis and the agent's last actuator output.  Integrating
    the remaining acceleration mismatch gives metres of potential lag.  This
    is a ranking heuristic; it must not be interpreted as a lower bound on
    true miss distance because attitude dynamics and disturbances are reduced.
    """
    if duration <= 0 or horizon <= 0:
        raise ValueError("Positive preview duration required")
    axis = np.asarray(body_axis, dtype=float)
    disturbance = np.asarray(disturbance, dtype=float)
    if (axis.shape != (3,) or disturbance.shape != (3,) or
            not np.all(np.isfinite(axis)) or
            not np.all(np.isfinite(disturbance)) or
            np.linalg.norm(axis) < 1e-9 or
            not np.isfinite(throttle) or
            min(axis_rate_limit, throttle_rate_limit) < 0):
        raise ValueError("Invalid actuator preview inputs")
    axis = axis / np.linalg.norm(axis)
    end = min(float(duration), float(horizon))
    times = np.linspace(0.0, end, 9)
    _, _, acceleration, _ = sample_curve(curve, duration, times)
    forces = acceleration - G - disturbance
    magnitudes = np.linalg.norm(forces, axis=1)
    limits = np.array([acceleration_limit(now + t) for t in times])
    maximum_throttle = np.clip(
        throttle + throttle_rate_limit * (times + response_allowance), 0, 1)
    maximum_force = limits * maximum_throttle
    cosines = np.sum(forces * axis, axis=1) / np.maximum(magnitudes, 1e-9)
    angle = np.arccos(np.clip(cosines, -1, 1))
    # Twice the planned axis-rate limit is an optimistic allowance for actual
    # body rotation; a positive shortfall is therefore a fairly strong signal.
    angle_gap = np.maximum(
        0.0, angle - 2.0 * axis_rate_limit * (times + response_allowance))
    directional = magnitudes * np.sin(np.minimum(angle_gap, np.pi / 2))
    radial = np.maximum(0.0, magnitudes - maximum_force)
    deficit = np.hypot(directional, radial)
    return float(np.trapezoid(deficit * (end - times), times))


class Scenario4ReachableRefreshAgent(Scenario4GuardedInnovationAgent):
    """Keep the route; relax only its timing when current actuation may lag."""

    def __init__(self, given_parameters, reach_min_tracking_error=0.9,
                 reach_min_shortfall=0.45, reach_max_remaining=6.0,
                 reach_delays=(0.0, 0.4, 0.8, 1.2), **kwargs):
        super().__init__(given_parameters, **kwargs)
        self.reach_min_tracking_error = float(reach_min_tracking_error)
        self.reach_min_shortfall = float(reach_min_shortfall)
        self.reach_max_remaining = float(reach_max_remaining)
        self.reach_delays = tuple(float(x) for x in reach_delays)
        if (not np.all(np.isfinite((self.reach_min_tracking_error,
                                   self.reach_min_shortfall,
                                   self.reach_max_remaining, *self.reach_delays))) or
                self.reach_min_tracking_error < 0 or
                self.reach_min_shortfall <= 0 or
                self.reach_max_remaining <= 0 or
                not self.reach_delays or
                any(x < 0 for x in self.reach_delays)):
            raise ValueError("Invalid reachable-refresh settings")
        self._reach_fallback = None
        self.diagnostics.update(reach_screened=0, reach_flagged=0,
                                reach_candidate_attempts=0,
                                reach_retained_nominal=0,
                                reach_refreshed=0,
                                reach_infeasible_joint_refresh=0,
                                reach_shortfall_sum=0.0)

    def _preview(self, curve, duration, now):
        body_axis = _rotate_body_to_world(self.quaternion, [0.0, 0.0, 1.0])
        return preview_actuator_shortfall(
            curve, duration, now, self.disturbance, body_axis,
            self.previous_throttle, self.available_acceleration,
            self.max_axis_rate, self.control['throttle_rate_limit'])

    def _accept_committed_curve(self, curve, duration, now):
        self._reach_fallback = None
        if not super()._accept_committed_curve(curve, duration, now):
            return False
        if (self.plan is None or self.filtered_position is None or
                duration < 1.5 or duration > self.reach_max_remaining):
            return True
        elapsed = np.clip(now - self.plan_start, 0, self.plan_duration)
        old_reference = sample_curve(self.plan, self.plan_duration, elapsed)[0]
        tracking_error = float(np.linalg.norm(
            old_reference - self.filtered_position))
        if tracking_error < self.reach_min_tracking_error:
            return True
        shortfall = self._preview(curve, duration, now)
        self.diagnostics['reach_screened'] += 1
        self.diagnostics['reach_shortfall_sum'] += shortfall
        if shortfall < self.reach_min_shortfall:
            return True
        self.diagnostics['reach_flagged'] += 1
        self._reach_fallback = (curve, duration, now, shortfall)
        return False

    def _rescue_committed_leg(self, observation, position, velocity, now,
                              duration):
        fallback = self._reach_fallback
        self._reach_fallback = None
        if not self.route:
            return False
        if fallback is None:
            # Nominal actuator/trajectory feasibility already failed.  Before
            # waiting on the stale command, release the old arrival velocity
            # and acceleration at the *same* target times.  This is a full-
            # route joint refresh, not an arbitrary timing extension.
            delays = (0.0,)
            nominal_shortfall = np.inf
        else:
            nominal_curve, nominal_duration, _, nominal_shortfall = fallback
            delays = self.reach_delays
        states = np.asarray(observation['balloon_states'], dtype=float)
        status = np.asarray(observation['balloon_status']).reshape(-1)
        route = list(self.route)
        times = np.diff(np.r_[now, self.deadlines])
        best = None
        cutoff = self.launch_time + self.burn_time - 1e-4
        if (len(times) == len(route) and np.all(times > 0.12) and
                np.all(status[route] == 1) and
                np.all(np.isfinite(states[route]))):
            for delay in delays:
                candidate_times = times.copy()
                candidate_times[0] += delay
                if now + np.sum(candidate_times) >= cutoff:
                    continue
                self.diagnostics['reach_candidate_attempts'] += 1
                arrivals = np.cumsum(candidate_times)
                targets = (states[route, :3] +
                           states[route, 3:6] * arrivals[:, None])
                try:
                    curves = free_chain(
                        position, velocity, self.acceleration, targets,
                        candidate_times, states[route[-1], 3:6],
                        self.terminal_weight)
                    starts = (now + np.r_[0.0, arrivals[:-1]])[:, None]
                    valid, _, ends_v, ends_a = self._valid(
                        curves, candidate_times, starts, samples=65)
                    if not np.all(valid):
                        continue
                    shortfall = self._preview(
                        curves[0], candidate_times[0], now)
                except (ValueError, np.linalg.LinAlgError,
                        FloatingPointError):
                    continue
                # The improved actuator preview must outweigh the cost of
                # postponing a moving target.  Preserve all downstream legs.
                if ((fallback is None or
                     shortfall + 0.15 + 0.2 * delay < nominal_shortfall) and
                        (best is None or shortfall + 0.2 * delay < best[0])):
                    best = (shortfall + 0.2 * delay, curves, candidate_times,
                            ends_v, ends_a)
        if best is not None:
            _, curves, candidate_times, ends_v, ends_a = best
            self.deadlines = list(now + np.cumsum(candidate_times))
            self.ends_v, self.ends_a = list(ends_v), list(ends_a)
            self._select(route[0], curves[0], float(candidate_times[0]), now)
            self.diagnostics['reach_refreshed'] += 1
            if fallback is None:
                self.diagnostics['reach_infeasible_joint_refresh'] += 1
            return True
        if fallback is None:
            return False
        # A failed preview intervention must not make the nominal refresh worse.
        self._select(route[0], nominal_curve, nominal_duration, now)
        self.diagnostics['reach_retained_nominal'] += 1
        return True
