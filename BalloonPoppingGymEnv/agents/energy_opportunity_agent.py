"""Observation-only fly-through guidance that preserves vertical momentum."""

import numpy as np

from BalloonPoppingGymEnv.agents.physics_guidance_agent import PhysicsGuidanceAgent


def constant_acceleration_intercept(position, velocity, target, duration):
    """Return a quintic-shaped coefficient array for a quadratic intercept."""
    duration = float(duration)
    if not np.isfinite(duration) or duration <= 0:
        raise ValueError("duration must be finite and positive")
    position = np.asarray(position, dtype=float)
    velocity = np.asarray(velocity, dtype=float)
    acceleration = 2.0 * (
        np.asarray(target, dtype=float) - position - velocity * duration
    ) / duration**2
    return np.stack(
        (
            position,
            velocity * duration,
            0.5 * acceleration * duration**2,
            np.zeros(3),
            np.zeros(3),
            np.zeros(3),
        )
    )


class EnergyOpportunityAgent(PhysicsGuidanceAgent):
    """Select reachable balloons without forcing a low-speed arrival.

    Each candidate is tested with a constant-acceleration intercept.  Unlike the
    minimum-jerk baseline, its terminal velocity is free, so a sequence of
    ascending balloons can retain the rocket's kinetic and potential energy.
    """

    def __init__(
        self,
        given_parameters,
        launch_time=36.3,
        replan_interval=0.2,
        minimum_duration=0.8,
        maximum_duration=6.0,
        duration_step=0.2,
        minimum_climb_acceleration=0.25,
        momentum_weight=0.08,
        **kwargs,
    ):
        super().__init__(
            given_parameters,
            launch_time=launch_time,
            replan_interval=replan_interval,
            **kwargs,
        )
        self.minimum_duration = float(minimum_duration)
        self.maximum_duration = float(maximum_duration)
        self.duration_step = float(duration_step)
        self.minimum_climb_acceleration = float(minimum_climb_acceleration)
        self.momentum_weight = float(momentum_weight)
        if self.minimum_duration <= 0 or self.maximum_duration < self.minimum_duration:
            raise ValueError("Invalid opportunity intercept duration range")
        if self.duration_step <= 0 or self.momentum_weight < 0:
            raise ValueError("Invalid opportunity search setting")
        self.diagnostics.update(
            opportunity_candidates=0,
            opportunity_feasible=0,
            opportunity_plans=0,
        )

    def _make_plan(self, observation, position, velocity, now):
        states = np.asarray(observation["balloon_states"], dtype=float)
        status = np.asarray(observation["balloon_status"]).reshape(-1)
        released = np.flatnonzero(status == 1)
        if not released.size:
            self.plan = None
            self.target_index = None
            return

        # A just-released balloon accelerates from rest; assigning the mature
        # cloud's drift immediately over-predicts its first few seconds.  The
        # frequent observation-only replan naturally incorporates its measured
        # velocity as it rises.
        drift = states[:, 3:6].copy()
        candidates = [
            int(index)
            for index in released
            if self.failed_until.get(int(index), 0.0) <= now
        ]
        locked = self.target_index in candidates
        if locked:
            candidates = [self.target_index]
        else:
            # Shortlist balloons already lying near the current velocity ray.
            lookahead = 2.5
            projected_rocket = position + velocity * lookahead
            candidates.sort(
                key=lambda index: np.linalg.norm(
                    states[index, :3] + drift[index] * lookahead - projected_rocket
                )
            )
            candidates = candidates[:24]

        best = None
        durations = np.arange(
            self.minimum_duration,
            self.maximum_duration + 0.5 * self.duration_step,
            self.duration_step,
        )
        remaining_burn = self.burn_time - (now - self.launch_time)
        for index in candidates:
            for duration in durations:
                if duration >= remaining_burn:
                    break
                target = states[index, :3] + drift[index] * duration
                curve = constant_acceleration_intercept(
                    position, velocity, target, duration
                )
                acceleration = 2.0 * curve[2] / duration**2
                terminal_velocity = velocity + acceleration * duration
                self.diagnostics["opportunity_candidates"] += 1
                if acceleration[2] < self.minimum_climb_acceleration:
                    continue
                if not self._feasible(curve, duration, now):
                    continue
                self.diagnostics["opportunity_feasible"] += 1
                cost = duration - self.momentum_weight * terminal_velocity[2]
                if best is None or cost < best[0]:
                    best = cost, duration, index, curve

        if best is None:
            self.diagnostics["no_feasible_plan"] += 1
            if self.plan is not None and now < self.plan_start + self.plan_duration:
                return
            if self.target_index is not None:
                self.failed_until[self.target_index] = now + 2 * self.replan_interval
            self.target_index = None
            self.plan = None
            return

        _, duration, index, curve = best
        if index != self.target_index:
            self.target_events.append((now, index))
        self.target_index = index
        self.plan, self.plan_start, self.plan_duration = curve, now, duration
        self.diagnostics["plans"] += 1
        self.diagnostics["opportunity_plans"] += 1
