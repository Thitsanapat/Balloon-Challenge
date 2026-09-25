"""Compare complete feasible group-entry routes against the original route.

Groups only generate alternative search starts. They do not contribute virtual
points to the final comparison: distinct planned hits and finish time determine
selection. The baseline is retained on ties, and failed group searches cannot
alter its controller state. All inputs are current released observations.
"""

import copy
import numpy as np
from BalloonPoppingGymEnv.agents.time_allocation_agent import TimeAllocationAgent


def observed_groups(states, released, position, velocity, radius, limit):
    """Bounded, overlapping coherent neighborhoods; never include hidden IDs."""
    ids = np.asarray(released, dtype=int)
    if len(ids) < 2 or limit <= 0:
        return []
    predicted = states[ids, :3]+states[ids, 3:6]*4.
    distance = np.linalg.norm(predicted[:, None]-predicted[None, :], axis=2)
    divergence = np.linalg.norm(states[ids, None, 3:6]-states[None, ids, 3:6], axis=2)
    near = (distance <= radius) & (divergence <= 4.)
    proposals = []
    for row in near:
        members = tuple(int(i) for i in ids[row])
        if len(members) < 2:
            continue
        entry = np.min(np.linalg.norm(predicted[row]-position-velocity*4., axis=1))
        # Ranking only limits work; the complete physical route search below
        # decides whether the travel and within-group turns are worthwhile.
        priority = len(members)/(1.+np.sqrt(entry))
        proposals.append((-priority, members))
    chosen = []
    for _, members in sorted(proposals):
        current = set(members)
        if any(len(current & set(old))/len(current | set(old)) >= .65 for old in chosen):
            continue
        chosen.append(members)
        if len(chosen) == limit:
            break
    return chosen


def portfolio_key(route, deadlines, original_count, original_finish, extra_only):
    """Return comparison key; equal-count routes can optionally be rejected."""
    count = len(set(route))
    if not count or count < original_count or (extra_only and count == original_count):
        return None
    finish = float(deadlines[-1])
    if count == original_count and finish >= original_finish-.1:
        return None
    return -count, finish


class GroupPortfolioAgent(TimeAllocationAgent):
    def __init__(self, given_parameters, group_radius=12., group_trials=2,
                 group_entry_depth=3, group_extra_only=True, **kwargs):
        super().__init__(given_parameters, **kwargs)
        self.group_radius = float(group_radius)
        self.group_trials = int(group_trials)
        self.group_entry_depth = int(group_entry_depth)
        self.group_extra_only = bool(group_extra_only)
        if (not np.isfinite(self.group_radius) or self.group_radius <= 0 or
                self.group_trials < 0 or self.group_entry_depth < 1):
            raise ValueError('Invalid group comparison settings')
        self.focus_ids = ()
        self.diagnostics.update(group_comparisons=0, group_adopted=0,
                                group_extra_planned_hits=0)

    def _candidate_ids(self, states, released, route, elapsed, position, velocity, now):
        if self.focus_ids and len(route) < self.group_entry_depth:
            members = [int(i) for i in released if i in self.focus_ids and i not in route
                       and self.failed_until.get(int(i), 0) <= now]
            if members:
                return super()._candidate_ids(states, members, route, elapsed, position, velocity, now)
        return super()._candidate_ids(states, released, route, elapsed, position, velocity, now)

    def _make_plan(self, observation, position, velocity, now):
        if not self.group_trials:
            return super()._make_plan(observation, position, velocity, now)
        before = copy.deepcopy(self.__dict__)
        count_events = len(self.route_events)
        super()._make_plan(observation, position, velocity, now)
        # Never disrupt final approach or an already committed reference.
        if len(self.route_events) == count_events or not self.route:
            return
        baseline = copy.deepcopy(self.__dict__)
        best_state = baseline
        original_count, original_finish = len(set(self.route)), self.deadlines[-1]
        best_key = (-original_count, original_finish)
        states = np.asarray(observation['balloon_states'], dtype=float)
        released = [int(i) for i in np.flatnonzero(np.asarray(
            observation['balloon_status']).reshape(-1) == 1)
            if self.failed_until.get(int(i), 0) <= now]
        groups = observed_groups(states, released, position, velocity, self.group_radius, self.group_trials)
        # Accumulate actual extra search work even for rejected proposals.
        work = {k: 0 for k in ('beam_searches', 'beam_candidates', 'beam_feasible', 'timing_solves')}
        comparisons = 0
        for members in groups:
            self.__dict__ = copy.deepcopy(before)
            self.focus_ids = members
            self.route, self.deadlines, self.ends_v, self.ends_a = [], [], [], []
            self.target_index, self.plan = None, None
            start_work = {k: self.diagnostics.get(k, 0) for k in work}
            super()._make_plan(observation, position, velocity, now)
            comparisons += 1
            for k in work:
                work[k] += self.diagnostics.get(k, 0)-start_work[k]
            key = portfolio_key(self.route, self.deadlines, original_count,
                                original_finish, self.group_extra_only)
            if key is not None and key < best_key:
                best_key = key
                best_state = copy.deepcopy(self.__dict__)
        self.__dict__ = best_state
        self.focus_ids = ()
        for k, value in work.items():
            self.diagnostics[k] = baseline['diagnostics'].get(k, 0)+value
        self.diagnostics['group_comparisons'] = baseline['diagnostics']['group_comparisons']+comparisons
        self.diagnostics['group_adopted'] = baseline['diagnostics']['group_adopted']+int(best_state is not baseline)
        self.diagnostics['group_extra_planned_hits'] = baseline['diagnostics']['group_extra_planned_hits']+len(set(self.route))-original_count
