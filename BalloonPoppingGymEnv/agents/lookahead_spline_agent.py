"""Two-intercept spline search ranked by a multi-target route heuristic."""

import numpy as np

from BalloonPoppingGymEnv.agents.spline_route_agent import (
    SplineRouteAgent, two_intercept_spline,
)


def greedy_tail_time(positions, start_index, excluded, count, speed):
    """Optimistic nearest-neighbour time used only to rank feasible prefixes."""
    remaining = [i for i in range(len(positions)) if i not in excluded]
    current = int(start_index)
    distance = 0.0
    for _ in range(min(int(count), len(remaining))):
        local = int(np.argmin([
            np.linalg.norm(positions[i] - positions[current]) for i in remaining
        ]))
        nxt = remaining.pop(local)
        distance += float(np.linalg.norm(positions[nxt] - positions[current]))
        current = nxt
    return distance / max(float(speed), 1e-9)


class LookaheadSplineAgent(SplineRouteAgent):
    def __init__(self, given_parameters, lookahead_targets=4,
                 lookahead_speed=25.0, lookahead_weight=1.0, **kwargs):
        super().__init__(given_parameters, **kwargs)
        self.lookahead_targets = int(lookahead_targets)
        self.lookahead_speed = float(lookahead_speed)
        self.lookahead_weight = float(lookahead_weight)
        if self.lookahead_targets < 0 or self.lookahead_speed <= 0 or self.lookahead_weight < 0:
            raise ValueError("Invalid route-lookahead settings")
        self.diagnostics.update(lookahead_plans=0, selected_route_cost=0.0)

    def _make_plan(self, observation, position, velocity, now):
        states = np.asarray(observation["balloon_states"], dtype=float)
        released = np.flatnonzero(
            np.asarray(observation["balloon_status"]).reshape(-1) == 1
        )
        moving = released[np.linalg.norm(states[released, 3:6], axis=1) > 0.5]
        shared = np.median(states[moving, 3:6], axis=0) if moving.size else np.zeros(3)
        drift = states[:, 3:6].copy()
        drift[np.linalg.norm(drift, axis=1) <= 0.5] = shared
        candidates = [int(i) for i in released if self.failed_until.get(int(i), 0) <= now]
        locked = self.target_index in candidates
        if locked:
            candidates = [self.target_index]
            eta = max(0.4, self.plan_start + self.plan_duration - now)
            first_times = np.array([eta, eta + 0.8, eta + 1.6])
        else:
            candidates.sort(key=lambda i: np.linalg.norm(states[i, :3] - position))
            candidates = candidates[:self.first_candidates]
            first_times = np.arange(2.0, self.horizon + 0.01, self.duration_step)

        remaining_burn = self.burn_time - (now - self.launch_time) - 1e-5
        best = None
        for first_time in first_times:
            if first_time >= remaining_burn:
                continue
            predicted_first = states[:, :3] + drift * first_time
            for first in candidates:
                others = [int(j) for j in released if j != first]
                others.sort(key=lambda j: np.linalg.norm(
                    predicted_first[j] - predicted_first[first]
                ))
                for second in others[:self.next_candidates]:
                    for second_time in np.arange(
                        1.5, self.second_horizon + 0.01, self.duration_step
                    ):
                        total = first_time + second_time
                        if total > remaining_burn:
                            break
                        self.diagnostics["pair_candidates"] += 1
                        projected = states[:, :3] + drift * total
                        first_position = states[first, :3] + drift[first] * first_time
                        c1, c2 = two_intercept_spline(
                            position, velocity, self.acceleration,
                            first_position, projected[second], drift[second],
                            first_time, second_time,
                        )
                        if not self._feasible(c1, first_time, now):
                            continue
                        if not self._feasible(c2, second_time, now + first_time):
                            continue
                        self.diagnostics["feasible_pairs"] += 1
                        released_positions = projected[released]
                        second_local = int(np.flatnonzero(released == second)[0])
                        excluded = {
                            int(np.flatnonzero(released == first)[0]), second_local
                        }
                        tail = greedy_tail_time(
                            released_positions, second_local, excluded,
                            self.lookahead_targets, self.lookahead_speed,
                        )
                        cost = total + self.lookahead_weight * tail
                        candidate = (cost, total, first, second, first_time, c1)
                        if best is None or candidate[:2] < best[:2]:
                            best = candidate
                        # Later times for the same pair cannot improve the prefix.
                        break

        if best is None:
            # Call the parent only as a safe fallback; it retains an active plan.
            super()._make_plan(observation, position, velocity, now)
            self.diagnostics["fallback_plans"] += 1
            return
        cost, total, first, second, duration, curve = best
        if first != self.target_index:
            self.target_events.append((now, first))
        self.route_events.append(
            [float(now), int(first), int(second), float(duration), float(total)]
        )
        self.target_index = first
        self.plan, self.plan_start, self.plan_duration = curve, now, float(duration)
        self.diagnostics["plans"] += 1
        self.diagnostics["pair_plans"] += 1
        self.diagnostics["lookahead_plans"] += 1
        self.diagnostics["selected_route_cost"] = float(cost)
