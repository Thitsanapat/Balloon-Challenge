"""Periodically compare the committed chain against a fresh observation plan.

Newly released balloons are allowed into the comparison. An inferior proposal
cannot overwrite the controller's existing plan or derivative estimates.
"""

import copy
import numpy as np
from BalloonPoppingGymEnv.agents.chain_launch_agent import ChainLaunchAgent


class RecedingChainAgent(ChainLaunchAgent):
    def __init__(self, given_parameters, route_review_interval=4.,
                 route_review_guard=1.5, route_review_time_gain=.5,
                 review_only_more_hits=True, **kwargs):
        super().__init__(given_parameters, **kwargs)
        self.route_review_interval = float(route_review_interval)
        self.route_review_guard = float(route_review_guard)
        self.route_review_time_gain = float(route_review_time_gain)
        self.review_only_more_hits = bool(review_only_more_hits)
        values = np.array([self.route_review_interval, self.route_review_guard,
                           self.route_review_time_gain])
        if np.any(~np.isfinite(values)) or np.any(values <= 0):
            raise ValueError('Finite positive route review settings required')
        self.next_route_review = 0.
        self.diagnostics.update(route_reviews=0, route_reviews_adopted=0,
                                route_reviews_extra_planned_hits=0)

    def _make_plan(self, observation, position, velocity, now):
        events = len(self.route_events)
        super()._make_plan(observation, position, velocity, now)
        if len(self.route_events) != events:
            self.next_route_review = now+self.route_review_interval
            return
        if (now < self.next_route_review or not self.route
                or self.deadlines[0]-now < self.route_review_guard):
            return
        self.next_route_review = now+self.route_review_interval
        status = np.asarray(observation['balloon_status']).reshape(-1)
        count = sum(status[i] == 1 for i in self.route)
        finish = self.deadlines[-1]
        # This is exclusively our own state, never a copy of the simulator.
        original = copy.deepcopy(self.__dict__)
        self.route, self.deadlines, self.ends_v, self.ends_a = [], [], [], []
        super()._make_plan(observation, position, velocity, now)
        candidate_count = len(self.route)
        better = bool(self.route) and (candidate_count > count or (
            not self.review_only_more_hits and candidate_count == count
            and self.deadlines[-1] <= finish-self.route_review_time_gain))
        # Preserve the total computational work even if a proposal is rejected.
        work = {key: self.diagnostics[key] for key in
                ('beam_searches', 'beam_candidates', 'beam_feasible', 'maximum_depth')}
        if not better:
            self.__dict__ = original
        self.diagnostics.update(work)
        self.diagnostics['route_reviews'] += 1
        if better:
            self.diagnostics['route_reviews_adopted'] += 1
            self.diagnostics['route_reviews_extra_planned_hits'] += int(candidate_count-count)
