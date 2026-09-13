"""Three-target-aware spline guidance with fly-through exit velocity.

Only two spline segments are solved, but the terminal velocity at the second
intercept points toward a third balloon.  This carries useful momentum through
the route without the cost of a full nonlinear three-intercept optimizer.
"""

import numpy as np

from BalloonPoppingGymEnv.agents.spline_route_agent import (
    SplineRouteAgent,
    two_intercept_spline,
)


class FlythroughSplineAgent(SplineRouteAgent):
    def __init__(self, given_parameters, flythrough_speed=5.0, **kwargs):
        super().__init__(given_parameters, **kwargs)
        self.flythrough_speed = float(flythrough_speed)
        if self.flythrough_speed < 0:
            raise ValueError("flythrough_speed must be non-negative")
        self.diagnostics.update(flythrough_plans=0)

    def _make_plan(self, observation, position, velocity, now):
        states = np.asarray(observation["balloon_states"], dtype=float)
        released = np.flatnonzero(
            np.asarray(observation["balloon_status"]).reshape(-1) == 1
        )
        moving = released[np.linalg.norm(states[released, 3:6], axis=1) > 0.5]
        shared = np.median(states[moving, 3:6], axis=0) if moving.size else np.zeros(3)
        drift = states[:, 3:6].copy()
        drift[np.linalg.norm(drift, axis=1) <= 0.5] = shared
        candidates = [
            int(i) for i in released if self.failed_until.get(int(i), 0) <= now
        ]
        locked = self.target_index in candidates
        if locked:
            candidates = [self.target_index]
            eta = max(0.4, self.plan_start + self.plan_duration - now)
            first_times = np.array([eta, eta + 0.8, eta + 1.6])
        else:
            candidates.sort(key=lambda i: np.linalg.norm(states[i, :3] - position))
            candidates = candidates[:self.first_candidates]
            first_times = np.arange(2.0, self.horizon + 0.01, self.duration_step)

        remaining = self.burn_time - (now - self.launch_time) - 1e-5
        best = None
        for first_time in first_times:
            if first_time >= remaining:
                continue
            if best is not None and first_time + 1.5 >= best[0]:
                continue
            predicted_first = states[:, :3] + drift * first_time
            for first in candidates:
                others = [int(j) for j in released if j != first]
                others.sort(
                    key=lambda j: np.linalg.norm(
                        predicted_first[j] - predicted_first[first]
                    )
                )
                for second in others[:self.next_candidates]:
                    for second_time in np.arange(
                        1.5, self.second_horizon + 0.01, self.duration_step
                    ):
                        total = first_time + second_time
                        if total > remaining or (best is not None and total >= best[0]):
                            break
                        projected = states[:, :3] + drift * total
                        thirds = [
                            int(j) for j in released if j != first and j != second
                        ]
                        terminal_velocity = drift[second].copy()
                        third = -1
                        if thirds and self.flythrough_speed > 0:
                            third = min(
                                thirds,
                                key=lambda j: np.linalg.norm(
                                    projected[j] - projected[second]
                                ),
                            )
                            delta = projected[third] - projected[second]
                            distance = float(np.linalg.norm(delta))
                            # v^2 <= a*d leaves roughly half the separation for
                            # braking/turning if the next choice changes.
                            available = self.available_acceleration(now + total)
                            lateral = np.sqrt(max(0.0, available**2 - 9.80665**2))
                            speed = min(
                                self.flythrough_speed,
                                np.sqrt(max(0.0, lateral * distance)),
                            )
                            terminal_velocity += speed * delta / max(distance, 1e-9)
                        self.diagnostics["pair_candidates"] += 1
                        c1, c2 = two_intercept_spline(
                            position,
                            velocity,
                            self.acceleration,
                            predicted_first[first],
                            projected[second],
                            terminal_velocity,
                            first_time,
                            second_time,
                        )
                        if not self._feasible(c1, first_time, now):
                            continue
                        if not self._feasible(c2, second_time, now + first_time):
                            continue
                        self.diagnostics["feasible_pairs"] += 1
                        best = total, first, second, third, first_time, c1
                        break

        if best is None:
            # Use the established safe planner if fly-through momentum makes
            # all of the bounded candidates infeasible.
            super()._make_plan(observation, position, velocity, now)
            self.diagnostics["fallback_plans"] += 1
            return
        total, first, second, third, duration, curve = best
        if first != self.target_index:
            self.target_events.append((now, first))
        self.route_events.append(
            [float(now), int(first), int(second), float(duration), float(total)]
        )
        self.target_index = first
        self.plan, self.plan_start, self.plan_duration = curve, now, float(duration)
        self.diagnostics["plans"] += 1
        self.diagnostics["pair_plans"] += 1
        self.diagnostics["flythrough_plans"] += 1
