"""Keep distinct feasible exit velocities in the joint-chain beam search.

Only current observations and given parameters are used. The frozen submission
agent remains the zero-bin control and retains the launch-axis comparison.
"""

import numpy as np

from BalloonPoppingGymEnv.agents.submission_time_allocation_v1 import (
    ChainBeamAgent, ChainSubmissionAgent, G, MomentumBeamAgent,
    boundary_curve,
)


class VelocityDiverseBeam(ChainBeamAgent):
    def __init__(self, given_parameters, exit_velocity_bin=4., pair_quota=3,
                 first_leg_min=2., **kwargs):
        super().__init__(given_parameters, **kwargs)
        self.exit_velocity_bin = float(exit_velocity_bin)
        self.pair_quota = int(pair_quota)
        self.first_leg_min = float(first_leg_min)
        if (not np.isfinite(self.exit_velocity_bin) or self.exit_velocity_bin < 0
                or not np.isfinite(pair_quota) or self.pair_quota != pair_quota
                or not 1 <= self.pair_quota <= 8
                or not np.isfinite(self.first_leg_min)
                or not .5 <= self.first_leg_min <= 2.):
            raise ValueError('Invalid bounded diverse-beam settings')
        self.diagnostics.update(diverse_searches=0, diverse_replaced_nodes=0,
                                diverse_velocity_bins=0)

    def _retain_diverse(self, children):
        """Fastest feasible child per exit-velocity bin and per pair quota."""
        children.sort(key=lambda node: self._node_priority(node, None, None, None))
        if self.exit_velocity_bin == 0:
            beam, counts = [], {}
            for node in children:
                pair = (node[0][0], node[0][-1])
                if counts.get(pair, 0) >= 2:
                    continue
                counts[pair] = counts.get(pair, 0) + 1
                beam.append(node)
                if len(beam) >= self.beam_width:
                    break
            return beam
        beam, pair_counts, bins = [], {}, set()
        for node in children:
            first_last = (node[0][0], node[0][-1])
            if pair_counts.get(first_last, 0) >= self.pair_quota:
                continue
            velocity = tuple(np.rint(node[4] / self.exit_velocity_bin).astype(int))
            key = first_last + (velocity,)
            if key in bins:
                continue
            bins.add(key)
            pair_counts[first_last] = pair_counts.get(first_last, 0) + 1
            beam.append(node)
            if len(beam) >= self.beam_width:
                break
        baseline, baseline_counts = [], {}
        for node in children:
            pair = (node[0][0], node[0][-1])
            if baseline_counts.get(pair, 0) >= 2:
                continue
            baseline_counts[pair] = baseline_counts.get(pair, 0) + 1
            baseline.append(node)
            if len(baseline) >= self.beam_width:
                break
        if [(n[0], n[1]) for n in beam] != [(n[0], n[1]) for n in baseline]:
            self.diagnostics['diverse_replaced_nodes'] += 1
        self.diagnostics['diverse_velocity_bins'] += len(bins)
        return beam

    def _make_plan(self, observation, position, velocity, now):
        # Exact parent behavior for the control and during provisional launch-axis
        # comparisons. Those comparisons restore a deep copy of controller state.
        if (self.exit_velocity_bin == 0 and self.first_leg_min == 2.) or not self.launch_selected:
            return ChainBeamAgent._make_plan(self, observation, position, velocity, now)

        states = np.asarray(observation['balloon_states'], dtype=float)
        released = np.flatnonzero(np.asarray(observation['balloon_status']).reshape(-1) == 1)
        drift = states[:, 3:6]
        while self.route and self.route[0] not in released:
            self.route.pop(0)
            self.deadlines.pop(0)
            self.ends_v.pop(0)
            self.ends_a.pop(0)
        if self.route and (self.commitment or self.route[0] == self.target_index):
            duration = self.deadlines[0] - now
            if duration > 0.12:
                target = self._intercept_target(states, self.route[0], duration)
                curve = boundary_curve(position, velocity, self.acceleration,
                                       target, self.ends_v[0], self.ends_a[0], duration)
                if self._feasible(curve, duration, now):
                    self._select(self.route[0], curve, duration, now)
                    return
                if self.plan is not None and now < self.plan_start + self.plan_duration:
                    return
            else:
                self.failed_until[self.route[0]] = now + 1.0
        remaining = self.launch_time + self.burn_time - now - 0.0001
        initial_force = self.acceleration - G - self.disturbance
        limit = 0.98 * self.available_acceleration(now)
        if np.linalg.norm(initial_force) > limit and limit > 0:
            self.acceleration = G + self.disturbance + initial_force * limit / np.linalg.norm(initial_force)
        beam = [((), (), None, position, velocity)]
        best = None
        self.diagnostics['beam_searches'] += 1
        self.diagnostics['diverse_searches'] += 1
        for depth in range(self.search_depth):
            children = []
            for route, times, _, end_p, end_v in beam:
                elapsed = sum(times)
                ids = self._candidate_ids(states, released, route, elapsed, end_p, end_v, now)
                first = self._first_leg_start(now)
                ts = np.arange(1.0 if depth else first,
                               min(self.leg_horizon, remaining - elapsed) + 0.001, self.time_grid)
                if depth and len(self.short_leg_times):
                    short = self.short_leg_times[self.short_leg_times <= min(self.leg_horizon, remaining - elapsed)]
                    ts = np.r_[short, ts]
                for index in ids:
                    for t in ts:
                        ids2, durations = route + (index,), times + (float(t),)
                        arrivals = np.cumsum(durations)
                        targets = states[list(ids2), :3] + drift[list(ids2)] * arrivals[:, None]
                        curves = self._chain_curves(position, velocity, targets, durations, drift[index])
                        starts = now + np.r_[0.0, arrivals[:-1]]
                        valid, ps, vs, _ = self._valid(curves, np.asarray(durations), starts[:, None], samples=17)
                        self.diagnostics['beam_candidates'] += 1
                        if not np.all(valid):
                            continue
                        self.diagnostics['beam_feasible'] += 1
                        children.append((ids2, durations, curves, ps[-1], vs[-1]))
            if not children:
                break
            beam = self._retain_diverse(children)
            if not beam:
                break
            best = beam[0]
            self.diagnostics['maximum_depth'] = max(self.diagnostics['maximum_depth'], depth + 1)
        if best is None:
            return MomentumBeamAgent._make_plan(self, observation, position, velocity, now)
        route, times, curves, _, _ = best
        valid, _, vs, acs = self._valid(curves, np.asarray(times),
            (now + np.r_[0.0, np.cumsum(times)[:-1]])[:, None], samples=65)
        if not np.all(valid):
            return MomentumBeamAgent._make_plan(self, observation, position, velocity, now)
        self.route, self.deadlines = list(route), list(now + np.cumsum(times))
        self.ends_v, self.ends_a = list(vs), list(acs)
        self.route_events.append([now, list(route), float(sum(times))])
        self._select(route[0], curves[0], times[0], now)

    def _first_leg_start(self, now):
        return self.first_leg_min if self.launched and now > self.launch_time + .5 else 2.


class VelocityDiverseAgent(ChainSubmissionAgent, VelocityDiverseBeam):
    """Keep the launch, final approach, and timing optimizer in the parent MRO."""
