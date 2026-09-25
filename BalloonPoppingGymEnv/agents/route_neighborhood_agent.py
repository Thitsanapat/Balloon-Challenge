"""Bounded online target-order search; no field, seed or replay access.

Mutations change target identities/order as well as arrival times. Every trial
jointly solves all waypoint derivatives and checks the reduced physical model.
This is a planning approximation, not a guarantee of a hit in the simulator.
"""

import numpy as np

from BalloonPoppingGymEnv.agents.chain_launch_agent import ChainLaunchAgent
from BalloonPoppingGymEnv.agents.chain_beam_agent import free_chain


def route_neighbors(route, durations, states, released, position, remaining, width=2):
    """Deterministic insertion, substitution, swap and relocation proposals.

    The geometric ranking only proposes candidates. It never accepts a route.
    Arrival coordinates use the current observation's constant-velocity model.
    """
    route, durations = tuple(route), tuple(float(t) for t in durations)
    arrivals = np.cumsum(durations)
    available = [int(i) for i in released if i not in route]
    targets = states[list(route), :3]+states[list(route), 3:6]*arrivals[:, None]
    # Insertion inside an existing leg preserves all other arrival times.
    for slot, duration in enumerate(durations):
        if duration < 1.:
            continue
        start = arrivals[slot]-duration
        origin = position if slot == 0 else targets[slot-1]
        midpoint = (origin+targets[slot])/2
        near = sorted(available, key=lambda i: np.linalg.norm(
            states[i, :3]+states[i, 3:6]*(start+duration/2)-midpoint))[:width]
        for index in near:
            for fraction in (.35, .5, .65):
                split = (duration*fraction, duration*(1-fraction))
                if min(split) >= .3:
                    yield route[:slot]+(index,)+route[slot:], durations[:slot]+split+durations[slot+1:]
    # Extend the tail if there is engine time left. Re-solving earlier legs can
    # change the exit state, so extension is not a fixed-state greedy append.
    for duration in (1., 2., 3.):
        if arrivals[-1]+duration > remaining:
            continue
        near = sorted(available, key=lambda i: np.linalg.norm(
            states[i, :3]+states[i, 3:6]*(arrivals[-1]+duration)-targets[-1]))[:width]
        for index in near:
            yield route+(index,), durations+(duration,)
    # Substitute nearby targets; this can enter a different downstream cluster.
    for slot in range(len(route)):
        near = sorted(available, key=lambda i: np.linalg.norm(
            states[i, :3]+states[i, 3:6]*arrivals[slot]-targets[slot]))[:width]
        for index in near:
            yield route[:slot]+(index,)+route[slot+1:], durations
    # Swap and relocate within a three-waypoint neighborhood, including the
    # first target. No waypoint identifiers are hard-coded.
    seen = {route}
    for left in range(len(route)):
        for right in range(left+1, min(len(route), left+4)):
            swap = list(route)
            swap[left], swap[right] = swap[right], swap[left]
            relocate = list(route)
            relocate.insert(right, relocate.pop(left))
            for proposal in (tuple(swap), tuple(relocate)):
                if proposal not in seen:
                    seen.add(proposal)
                    yield proposal, durations
    # A timing-only neighbor also permits a shorter route to free tail time.
    yield route, durations


class RouteNeighborhoodAgent(ChainLaunchAgent):
    def __init__(self, given_parameters, neighborhood_budget=1200,
                 neighborhood_rounds=3, neighborhood_beam=4,
                 neighborhood_width=2, only_more_hits=True, **kwargs):
        super().__init__(given_parameters, **kwargs)
        self.neighborhood_budget = int(neighborhood_budget)
        self.neighborhood_rounds = int(neighborhood_rounds)
        self.neighborhood_beam = int(neighborhood_beam)
        self.neighborhood_width = int(neighborhood_width)
        self.only_more_hits = bool(only_more_hits)
        if self.neighborhood_budget < 0 or min(self.neighborhood_rounds,
                self.neighborhood_beam, self.neighborhood_width) < 1:
            raise ValueError('Invalid neighborhood search limits')
        self.diagnostics.update(neighborhood_searches=0, neighborhood_trials=0,
                                neighborhood_feasible=0, neighborhood_adopted=0,
                                neighborhood_extra_planned_hits=0)

    def _solve_route(self, route, durations, states, position, velocity, now, samples):
        arrivals = np.cumsum(durations)
        targets = states[list(route), :3]+states[list(route), 3:6]*arrivals[:, None]
        curves = free_chain(position, velocity, self.acceleration, targets,
                            durations, states[route[-1], 3:6], self.terminal_weight)
        valid, _, vs, acs = self._valid(curves, np.asarray(durations),
                                     (now+np.r_[0., arrivals[:-1]])[:, None], samples=samples)
        if np.all(valid):
            return curves, vs, acs
        return None

    def _make_plan(self, observation, position, velocity, now):
        events = len(self.route_events)
        super()._make_plan(observation, position, velocity, now)
        if not self.neighborhood_budget or not self.route or len(self.route_events) == events:
            return
        states = np.asarray(observation['balloon_states'], dtype=float)
        released = np.flatnonzero(np.asarray(observation['balloon_status']).reshape(-1) == 1)
        released = [int(i) for i in released if self.failed_until.get(int(i), 0) <= now]
        original = tuple(self.route)
        times = tuple(np.diff(np.r_[now, self.deadlines]))
        remaining = self.launch_time+self.burn_time-now-1e-4
        if min(times) < .15 or not set(original).issubset(released):
            return
        self.diagnostics['neighborhood_searches'] += 1
        priority = lambda node: (-len(node[0]), sum(node[1]))
        initial = (original, times)
        frontier, elites = [initial], [initial]
        visited = {(original, tuple(np.round(times, 6)))}
        trials = 0
        for _ in range(self.neighborhood_rounds):
            candidates = []
            for route, durations in frontier:
                for trial_route, nominal in route_neighbors(route, durations, states,
                        released, position, remaining, self.neighborhood_width):
                    for scale in (1., .95, 1.05):
                        ts = tuple(float(t*scale) for t in nominal)
                        key = (trial_route, tuple(np.round(ts, 6)))
                        if key in visited or sum(ts) > remaining or min(ts) < .25:
                            continue
                        if trials >= self.neighborhood_budget:
                            break
                        visited.add(key)
                        trials += 1
                        self.diagnostics['neighborhood_trials'] += 1
                        solved = self._solve_route(trial_route, ts, states, position, velocity, now, 17)
                        if solved is not None:
                            self.diagnostics['neighborhood_feasible'] += 1
                            candidates.append((trial_route, ts))
                    if trials >= self.neighborhood_budget:
                        break
                if trials >= self.neighborhood_budget:
                    break
            if not candidates:
                break
            candidates.sort(key=priority)
            elites.extend(candidates)
            # One timing per sequence preserves target-order diversity.
            sequences, frontier = set(), []
            for candidate in candidates:
                if candidate[0] in sequences:
                    continue
                sequences.add(candidate[0])
                frontier.append(candidate)
                if len(frontier) >= self.neighborhood_beam:
                    break
            if trials >= self.neighborhood_budget:
                break
        for route, durations in sorted(elites, key=priority):
            if priority((route, durations)) >= priority(initial):
                break
            if self.only_more_hits and len(route) <= len(original):
                break
            solved = self._solve_route(route, durations, states, position, velocity, now, 65)
            if solved is None:
                continue
            curves, vs, acs = solved
            self.route, self.deadlines = list(route), list(now+np.cumsum(durations))
            self.ends_v, self.ends_a = list(vs), list(acs)
            self.route_events[-1] = [now, list(route), float(sum(durations))]
            self._select(route[0], curves[0], durations[0], now)
            self.diagnostics['neighborhood_adopted'] += 1
            self.diagnostics['neighborhood_extra_planned_hits'] += len(route)-len(original)
            break
