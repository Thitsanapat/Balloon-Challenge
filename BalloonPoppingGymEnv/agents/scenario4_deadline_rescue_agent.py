"""Recover a live Scenario 4 intercept by modestly retiming its first leg.

Only the current observation, observed wind profile, and public rocket model
are used. A replacement must pass the existing dense trajectory checks with
the configured thrust reserve. If no replacement passes, the parent keeps its
previous behavior and plan.
"""

import numpy as np

from BalloonPoppingGymEnv.agents.constrained_chain_agent import repair_margins
from BalloonPoppingGymEnv.agents.scenario4_wind_profile_agent import (
    Scenario4WindProfileAgent,
)
from BalloonPoppingGymEnv.agents.submission_sqp_v2 import (
    G, boundary_curve, sample_curve,
)


class Scenario4DeadlineRescueAgent(Scenario4WindProfileAgent):
    """Try a nearby deadline when a committed-leg refresh is infeasible."""

    def __init__(self, given_parameters, rescue_max_remaining=6.0,
                 rescue_trigger_radius=1.0,
                 rescue_offsets=(0.0, 0.25, 0.5, 1.0, 1.5, 2.0), **kwargs):
        super().__init__(given_parameters, **kwargs)
        self.rescue_max_remaining = float(rescue_max_remaining)
        self.rescue_trigger_radius = float(rescue_trigger_radius)
        self.rescue_offsets = np.asarray(rescue_offsets, dtype=float)
        if (not np.isfinite(self.rescue_max_remaining) or
                self.rescue_max_remaining <= 0 or
                not np.isfinite(self.rescue_trigger_radius) or
                self.rescue_trigger_radius <= 0 or
                self.rescue_offsets.ndim != 1 or not len(self.rescue_offsets) or
                not np.all(np.isfinite(self.rescue_offsets)) or
                np.any(self.rescue_offsets < 0)):
            raise ValueError("Invalid deadline rescue settings")
        self.rescue_offsets = np.unique(self.rescue_offsets)
        self.diagnostics.update(committed_refresh_failed=0,
                                deadline_rescue_attempts=0,
                                deadline_rescue_accepted=0)

    def _rescue_committed_leg(self, observation, position, velocity, now, duration):
        self.diagnostics['committed_refresh_failed'] += 1
        if (duration > self.rescue_max_remaining or self.plan is None or
                not self.route or self.target_index != self.route[0] or
                now >= self.plan_start + self.plan_duration):
            return False

        index = self.route[0]
        states = np.asarray(observation['balloon_states'], dtype=float)
        if not np.all(np.isfinite(states[index])):
            return False
        current_target = states[index, :3] + states[index, 3:6] * duration
        old_endpoint = sample_curve(self.plan, self.plan_duration,
                                    self.plan_duration)[0]
        if (not np.all(np.isfinite(old_endpoint)) or
                np.linalg.norm(old_endpoint - current_target) <=
                self.radius * self.rescue_trigger_radius):
            return False

        burn_remaining = self.launch_time + self.burn_time - now - 1e-4
        maximum = min(self.leg_horizon, burn_remaining)
        if maximum < max(duration, 0.3):
            return False
        self.diagnostics['deadline_rescue_attempts'] += 1

        # Match the chain search's initial-force clipping locally. The state
        # estimate itself is left untouched if every candidate fails.
        start_a = np.asarray(self.acceleration, dtype=float).copy()
        initial_force = start_a - G - self.disturbance
        limit = .98 * self.available_acceleration(now)
        magnitude = float(np.linalg.norm(initial_force))
        if limit > 0 and magnitude > limit:
            start_a = G + self.disturbance + initial_force * limit / magnitude

        drift = states[index, 3:6]
        old_v = np.asarray(self.ends_v[0], dtype=float)
        old_a = np.asarray(self.ends_a[0], dtype=float)
        velocities = (drift, .5 * (drift + old_v), old_v)
        accelerations = (np.zeros(3), old_a)
        for offset in self.rescue_offsets:
            trial = float(duration + offset)
            if trial < .3 or trial > maximum:
                continue
            target = states[index, :3] + drift * trial
            for end_v in velocities:
                for end_a in accelerations:
                    candidate = boundary_curve(position, velocity, start_a, target,
                                               end_v, end_a, trial)
                    durations = np.array([trial])
                    valid, _, vs, acs = self._valid(
                        candidate[None, :, :], durations,
                        np.array([[now]]), samples=65)
                    if not bool(valid[0]):
                        continue
                    margins = repair_margins(
                        self, candidate[None, :, :], durations, now, 65, 0.)
                    if not np.all(np.isfinite(margins)) or np.min(margins) < -1e-8:
                        continue
                    self.route = [index]
                    self.deadlines = [now + trial]
                    self.ends_v = [vs[0]]
                    self.ends_a = [acs[0]]
                    self._select(index, candidate, trial, now)
                    self.diagnostics['deadline_rescue_accepted'] += 1
                    return True
        return False
