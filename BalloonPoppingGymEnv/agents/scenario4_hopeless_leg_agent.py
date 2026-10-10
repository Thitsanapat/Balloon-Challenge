"""Escape a committed intercept only when its observed gap is out of reach.

The normal Scenario 4 planner keeps the old spline when a same-target refresh
fails.  This experiment checks a generous horizontal reach envelope using the
current rocket estimate, observed balloon state, and public thrust limits.  It
tries a short alternative beam search only after that screen fails, and keeps
the original plan if no alternative is found.  This is a planning heuristic,
not a claim that future wind or aerodynamic acceleration is known exactly.
"""

import copy

import numpy as np

from BalloonPoppingGymEnv.agents.chain_beam_agent import ChainBeamAgent
from BalloonPoppingGymEnv.agents.scenario4_wind_profile_agent import (
    Scenario4WindProfileAgent,
)


def optimistic_horizontal_miss_lower_bound(relative_position,
                                            relative_velocity, horizon,
                                            acceleration_bound,
                                            samples=25):
    """Lower bound on horizontal separation with generous relative control.

    The rocket may choose any horizontal acceleration up to ``acceleration_bound``
    at every instant.  The target's unmodelled acceleration is included in that
    bound by the caller.  Sampling the entire interval permits an early pass;
    checking only the planned endpoint would incorrectly reject such passes.
    """
    position = np.asarray(relative_position, dtype=float)
    velocity = np.asarray(relative_velocity, dtype=float)
    horizon = float(horizon)
    acceleration_bound = float(acceleration_bound)
    samples = int(samples)
    if (position.shape != (2,) or velocity.shape != (2,) or
            not np.all(np.isfinite(position)) or
            not np.all(np.isfinite(velocity)) or
            not np.isfinite(horizon) or horizon <= 0 or
            not np.isfinite(acceleration_bound) or acceleration_bound < 0 or
            samples < 3):
        raise ValueError("Invalid horizontal reachability input")
    times = np.linspace(0.0, horizon, samples)
    ballistic_gap = np.linalg.norm(
        position[None, :] + times[:, None] * velocity[None, :], axis=1)
    reachable_shift = 0.5 * acceleration_bound * times**2
    # Sampling alone could miss a brief near-pass between samples.  The
    # derivative of this lower envelope is bounded by |v| + a*horizon, so
    # subtract the worst gap to the nearest sample before making a decision.
    interval = horizon / (samples - 1)
    interpolation_allowance = (
        np.linalg.norm(velocity) + acceleration_bound * horizon
    ) * interval / 2.0
    return float(max(0.0, np.min(ballistic_gap - reachable_shift) -
                     interpolation_allowance))


class Scenario4HopelessLegAgent(Scenario4WindProfileAgent):
    """Try a new observed target after a failed and clearly stale refresh."""

    def __init__(self, given_parameters, escape_max_remaining=4.0,
                 escape_clearance=2.0, escape_rocket_acceleration_slack=3.0,
                 escape_target_acceleration_slack=4.0,
                 escape_cooldown=1.5, escape_retry_interval=0.8,
                 escape_depth=3, escape_beam_width=8,
                 escape_branch_targets=8, **kwargs):
        super().__init__(given_parameters, **kwargs)
        self.escape_max_remaining = float(escape_max_remaining)
        self.escape_clearance = float(escape_clearance)
        self.escape_rocket_acceleration_slack = float(
            escape_rocket_acceleration_slack)
        self.escape_target_acceleration_slack = float(
            escape_target_acceleration_slack)
        self.escape_cooldown = float(escape_cooldown)
        self.escape_retry_interval = float(escape_retry_interval)
        self.escape_depth = int(escape_depth)
        self.escape_beam_width = int(escape_beam_width)
        self.escape_branch_targets = int(escape_branch_targets)
        values = (self.escape_max_remaining, self.escape_clearance,
                  self.escape_rocket_acceleration_slack,
                  self.escape_target_acceleration_slack,
                  self.escape_cooldown, self.escape_retry_interval)
        if (not np.all(np.isfinite(values)) or values[0] <= 0 or
                min(values[1:]) < 0 or
                min(self.escape_depth, self.escape_beam_width,
                    self.escape_branch_targets) < 1):
            raise ValueError("Invalid hopeless-leg escape settings")
        self._escape_retry_after = {}
        self.diagnostics.update(escape_refresh_failures=0,
                                escape_out_of_reach=0,
                                escape_alternative_attempts=0,
                                escape_alternative_accepted=0)

    def _rescue_committed_leg(self, observation, position, velocity, now,
                              duration):
        self.diagnostics["escape_refresh_failures"] += 1
        if (duration <= 0 or duration > self.escape_max_remaining or
                not self.route or self.plan is None or
                self.target_index != self.route[0] or
                now < self._escape_retry_after.get(self.route[0], -np.inf) or
                now + 0.5 >= self.launch_time + self.burn_time):
            return False
        states = np.asarray(observation["balloon_states"], dtype=float)
        status = np.asarray(observation["balloon_status"]).reshape(-1)
        index = int(self.route[0])
        alternatives = [int(i) for i in np.flatnonzero(status == 1)
                        if i != index and
                        self.failed_until.get(int(i), 0.0) <= now]
        if (not alternatives or status[index] != 1 or
                not np.all(np.isfinite(states[index]))):
            return False

        maximum_thrust_acceleration = max(
            self.available_acceleration(now),
            self.available_acceleration(now + duration))
        # Allow any horizontal steering direction, plus extra disturbance and
        # target acceleration.  This is intentionally favorable to keeping the
        # current target, especially early in the leg.
        horizontal_bound = (
            maximum_thrust_acceleration * np.sin(self.max_tilt) +
            self.escape_rocket_acceleration_slack +
            self.escape_target_acceleration_slack)
        gap = optimistic_horizontal_miss_lower_bound(
            states[index, :2] - position[:2],
            states[index, 3:5] - velocity[:2],
            duration, horizontal_bound)
        if gap <= self.radius + self.escape_clearance:
            return False
        self.diagnostics["escape_out_of_reach"] += 1
        self.diagnostics["escape_alternative_attempts"] += 1
        self._escape_retry_after[index] = now + self.escape_retry_interval

        # Run a limited alternate search against the already-adjusted observed
        # balloon states.  Calling the wind-profile wrapper again would apply
        # its correction twice.  Roll back every changed field if no feasible
        # alternate route is produced.
        original = copy.deepcopy(self.__dict__)
        original_limits = (self.search_depth, self.beam_width,
                           self.branch_targets)
        self.failed_until[index] = max(
            self.failed_until.get(index, 0.0), now + self.escape_cooldown)
        self.route, self.deadlines = [], []
        self.ends_v, self.ends_a = [], []
        self.target_index = None
        self.plan = None
        self.search_depth = min(self.search_depth, self.escape_depth)
        self.beam_width = min(self.beam_width, self.escape_beam_width)
        self.branch_targets = min(self.branch_targets,
                                  self.escape_branch_targets)
        try:
            ChainBeamAgent._make_plan(self, observation, position, velocity,
                                       now)
            accepted = bool(self.route and self.route[0] != index and
                            self.plan is not None)
        except (ValueError, np.linalg.LinAlgError, FloatingPointError):
            accepted = False
        if not accepted:
            self.__dict__ = original
            return False
        self.search_depth, self.beam_width, self.branch_targets = original_limits
        self.diagnostics["escape_alternative_accepted"] += 1
        return True
