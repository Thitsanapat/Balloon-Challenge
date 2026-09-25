"""Bounded paired-successor proposals around a frozen feasible incumbent.

A rejected minimum-jerk prefix is not a proof that every longer chain through
those waypoints is infeasible: free_chain reoptimizes all endpoint derivatives.
Here a rejected prefix gets ONE extra successor. Only the recomputed, fully
validated chain can become executable. No simulator state or stored fields.
"""

import numpy as np

from BalloonPoppingGymEnv.agents.submission_time_allocation_v1 import ChainSubmissionAgent


class BridgeChainAgent(ChainSubmissionAgent):
    def __init__(self, given_parameters, bridge_budget=256, bridge_width=3,
                 bridge_prefixes=3, bridge_branch_targets=3,
                 bridge_first_leg_min=2., bridge_allow_faster=False, **kwargs):
        super().__init__(given_parameters, **kwargs)
        settings = ((bridge_budget, 0, 4096), (bridge_width, 1, 8),
                    (bridge_prefixes, 1, 8), (bridge_branch_targets, 1, 8))
        for value, lower, upper in settings:
            if not np.isfinite(value) or int(value) != value or not lower <= value <= upper:
                raise ValueError('Bridge search limits must be bounded integers')
        self.bridge_budget = int(bridge_budget)
        self.bridge_width = int(bridge_width)
        self.bridge_prefixes = int(bridge_prefixes)
        self.bridge_branch_targets = int(bridge_branch_targets)
        self.bridge_first_leg_min = float(bridge_first_leg_min)
        if not np.isfinite(self.bridge_first_leg_min) or not .5 <= self.bridge_first_leg_min <= 2.:
            raise ValueError('Bridge first-leg minimum must be finite in [.5, 2]')
        self.bridge_allow_faster = bool(bridge_allow_faster)
        self.bridge_events = []
        self.diagnostics.update(bridge_searches=0, bridge_candidates=0,
                                bridge_rejected_prefixes=0, bridge_recovered=0,
                                bridge_dense_rejected=0, bridge_adopted=0,
                                bridge_extra_planned_hits=0)

    def _bridge_times(self, prefix, remaining, now):
        lower = 1.
        if not prefix:
            lower = self.bridge_first_leg_min if self.launched and now > self.launch_time + .5 else 2.
        limit = min(self.leg_horizon, remaining)
        times = np.arange(lower, limit + .001, self.time_grid)
        if prefix and len(self.short_leg_times):
            times = np.r_[self.short_leg_times[self.short_leg_times <= limit], times]
        return times[times <= limit+1e-10]

    def _bridge_probes(self, prefix, remaining, now, allowance, target_count):
        """Cover the whole duration grid within a fixed number of fits.

        Earliest-only truncation can discard every feasible long first intercept.
        Keep both endpoints whenever at least two duration slots fit, and include
        the incumbent first-leg grid neighbor when an interior slot is available.
        """
        grid = np.asarray(self._bridge_times(prefix, remaining, now), dtype=float)
        if not allowance or not target_count or not len(grid):
            return grid[:0]
        slots = min(len(grid), max(1, int(np.ceil(allowance/target_count))))
        selected = np.rint(np.linspace(0, len(grid)-1, slots)).astype(int)
        if not prefix and slots >= 3 and self.deadlines:
            preferred = int(np.argmin(np.abs(grid-(self.deadlines[0]-now))))
            if preferred not in selected:
                interior = 1 + int(np.argmin(np.abs(selected[1:-1]-preferred)))
                selected[interior] = preferred
        return grid[np.sort(selected)]

    def _bridge_search(self, states, released, position, velocity, now):
        """Return a dense-checked improvement, or None; do not mutate the plan.

        Each incumbent seed is an executable prefix of the baseline's complete curve.
        Recomputed prefix fits need not remain feasible, so they may be extended
        once. Recovered feasible nodes can continue expanding. Root receives half
        the budget; the other half is shared across incumbent-prefix alternatives.
        """
        remaining = self.launch_time + self.burn_time - now - .0001
        route = tuple(self.route)
        times = tuple(np.diff(np.r_[now, self.deadlines]))
        usable = (len(route) == len(times) and len(route) > 0
                  and np.all(np.isfinite(times)) and min(times) > 0
                  and sum(times) <= remaining and set(route).issubset(set(released)))
        seeds = []
        if usable:
            for length in range(len(route), max(0, len(route)-self.bridge_prefixes), -1):
                index = route[length-1]
                elapsed = sum(times[:length])
                end_p = states[index, :3] + states[index, 3:6]*elapsed
                seeds.append((route[:length], times[:length], end_p, self.ends_v[length-1]))
        seeds.append(((), (), position, velocity))
        incumbent_key = (-len(route), sum(times)) if usable else (0, float('inf'))
        best, best_key = None, incumbent_key
        used = 0
        if len(seeds) == 1:
            quotas = [self.bridge_budget]
        else:
            tail_budget = self.bridge_budget//2
            quotas = [tail_budget//(len(seeds)-1)]*(len(seeds)-1)
            for index in range(tail_budget % (len(seeds)-1)):
                quotas[index] += 1
            quotas.append(self.bridge_budget-tail_budget)

        def fit(ids, durations):
            nonlocal used
            used += 1
            self.diagnostics['bridge_candidates'] += 1
            arrivals = np.cumsum(durations)
            targets = states[list(ids), :3] + states[list(ids), 3:6]*arrivals[:, None]
            curves = self._chain_curves(position, velocity, targets, durations, states[ids[-1], 3:6])
            starts = (now + np.r_[0., arrivals[:-1]])[:, None]
            valid, ps, vs, _ = self._valid(curves, np.asarray(durations), starts, samples=17)
            return (ids, durations, curves, ps[-1], vs[-1]), valid

        def consider(node, recovered):
            nonlocal best, best_key
            ids, durations, curves, _, _ = node
            key = (-len(ids), sum(durations))
            if key >= best_key or (not self.bridge_allow_faster and len(ids) <= -incumbent_key[0]):
                return
            starts = (now + np.r_[0., np.cumsum(durations)[:-1]])[:, None]
            valid, _, vs, acs = self._valid(curves, np.asarray(durations), starts, samples=65)
            if not np.all(valid):
                self.diagnostics['bridge_dense_rejected'] += 1
                return
            if recovered:
                self.diagnostics['bridge_recovered'] += 1
            best, best_key = (ids, durations, curves, vs, acs), key

        for seed, quota in zip(seeds, quotas):
            prefix, prefix_times, end_p, end_v = seed
            begin = used
            frontier = [(prefix, prefix_times, None, end_p, end_v)]
            # Reserve capacity for later layers rather than exhausting the budget
            # on all time choices at the first waypoint. This is a heuristic cap,
            # never a weakening of feasibility checks.
            block = max(8, quota//max(1, self.search_depth-len(prefix)))
            while frontier and used-begin < quota:
                prefix, prefix_times, _, end_p, end_v = frontier.pop(0)
                if len(prefix) >= self.search_depth:
                    continue
                block_start = used
                block_limit = min(quota-(used-begin), block)
                first_limit = max(1, block_limit//2)
                elapsed = sum(prefix_times)
                ids = self._candidate_ids(states, released, prefix, elapsed, end_p, end_v, now)
                ids = ids[:self.bridge_branch_targets]
                rejects, feasible = [], []
                # Interleave targets at each duration instead of starving targets.
                probes = self._bridge_probes(prefix, remaining-elapsed, now, first_limit, len(ids))
                for duration in probes:
                    for index in ids:
                        if used-block_start >= first_limit:
                            break
                        child, valid = fit(prefix+(int(index),), prefix_times+(float(duration),))
                        if np.all(valid):
                            feasible.append(child)
                            consider(child, False)
                        else:
                            self.diagnostics['bridge_rejected_prefixes'] += 1
                            rejects.append((int(np.count_nonzero(~valid)), child))
                    if used-block_start >= first_limit:
                        break
                rejects.sort(key=lambda item: (item[0], sum(item[1][1]), item[1][0]))
                # No invalid child enters frontier: only one recovery successor.
                for _, child in rejects[:self.bridge_width]:
                    failed_route, failed_times, _, failed_p, failed_v = child
                    if len(failed_route) >= self.search_depth:
                        continue
                    elapsed = sum(failed_times)
                    ids = self._candidate_ids(states, released, failed_route, elapsed, failed_p, failed_v, now)
                    ids = ids[:self.bridge_branch_targets]
                    probes = self._bridge_probes(failed_route, remaining-elapsed, now,
                                                 block_limit-(used-block_start), len(ids))
                    for duration in probes:
                        for index in ids:
                            if used-block_start >= block_limit:
                                break
                            extended, valid = fit(failed_route+(int(index),), failed_times+(float(duration),))
                            if np.all(valid):
                                feasible.append(extended)
                                consider(extended, True)
                        if used-block_start >= block_limit:
                            break
                    if used-block_start >= block_limit:
                        break
                frontier.extend(feasible)
                frontier.sort(key=lambda node: (-len(node[0]), sum(node[1]), node[0]))
                # Deterministic bounded frontier; never retain duplicate routes
                # with identical timing. Different timing hypotheses remain useful.
                unique, seen = [], set()
                for node in frontier:
                    key = (node[0], node[1])
                    if key not in seen:
                        unique.append(node)
                        seen.add(key)
                    if len(unique) >= self.bridge_width:
                        break
                frontier = unique
        return best

    def _make_plan(self, observation, position, velocity, now):
        if self.bridge_budget == 0:
            return super()._make_plan(observation, position, velocity, now)
        searches = self.diagnostics['beam_searches']
        super()._make_plan(observation, position, velocity, now)
        # The parent probes launch axes on temporary deep-copied state. Extra
        # proposals there can change the chosen axis while their diagnostics
        # disappear on restore. Keep launch selection identical to the incumbent;
        # search only after that public-observation-based choice is committed.
        if not self.launch_selected:
            return
        # Do not disrupt commitment or a final-approach hold.
        if self.diagnostics['beam_searches'] == searches:
            return
        states = np.asarray(observation['balloon_states'], dtype=float)
        released = np.flatnonzero(np.asarray(observation['balloon_status']).reshape(-1) == 1)
        if not len(released) or not np.all(np.isfinite(states[released])):
            return
        self.diagnostics['bridge_searches'] += 1
        previous_count = len(self.route)
        result = self._bridge_search(states, released, position, velocity, now)
        if result is None:
            return
        ids, durations, curves, vs, acs = result
        self.route, self.deadlines = list(ids), list(now + np.cumsum(durations))
        self.ends_v, self.ends_a = list(vs), list(acs)
        event = [now, list(ids), float(sum(durations))]
        if self.route_events and self.route_events[-1][0] == now:
            self.route_events[-1] = event
        else:
            self.route_events.append(event)
        self._select(ids[0], curves[0], durations[0], now)
        self.bridge_events.append({'time': float(now), 'first_leg': float(durations[0]),
                                   'durations': list(durations), 'planned_hits': len(ids),
                                   'previous_planned_hits': previous_count})
        self.diagnostics['bridge_adopted'] += 1
        self.diagnostics['bridge_extra_planned_hits'] += len(ids)-previous_count
