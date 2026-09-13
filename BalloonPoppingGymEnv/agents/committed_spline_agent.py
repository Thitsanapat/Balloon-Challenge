"""Low-cost receding-horizon spline guidance with a committed next target.

The parent planner already evaluates two consecutive intercepts.  This variant
keeps the second intercept as a commitment, so a pop promotes it immediately
instead of restarting an unconstrained target search.  The commitment is
refreshed while approaching the current target and is abandoned only when it
is no longer released or no feasible continuation can be found.
"""

import numpy as np

from BalloonPoppingGymEnv.agents.spline_route_agent import SplineRouteAgent


class CommittedSplineAgent(SplineRouteAgent):
    def __init__(self, given_parameters, **kwargs):
        super().__init__(given_parameters, **kwargs)
        self.committed_target = None
        self.diagnostics.update(
            committed_promotions=0,
            committed_fallbacks=0,
            commitment_updates=0,
        )

    def _plan_once(self, observation, position, velocity, now, forced_target=None):
        if forced_target is None:
            super()._make_plan(observation, position, velocity, now)
            return
        released = np.flatnonzero(
            np.asarray(observation["balloon_status"]).reshape(-1) == 1
        )
        saved = self.failed_until.copy()
        try:
            for index in released:
                if int(index) != int(forced_target):
                    self.failed_until[int(index)] = np.inf
            super()._make_plan(observation, position, velocity, now)
        finally:
            self.failed_until = saved

    def _make_plan(self, observation, position, velocity, now):
        status = np.asarray(observation["balloon_status"]).reshape(-1)
        promoted = None
        if self.target_index is None and self.committed_target is not None:
            if status[self.committed_target] == 1:
                promoted = int(self.committed_target)
                self.diagnostics["committed_promotions"] += 1
            else:
                self.committed_target = None

        event_count = len(self.route_events)
        self._plan_once(observation, position, velocity, now, promoted)
        if promoted is not None and self.plan is None:
            # A stale commitment must never strand the vehicle.  Retry the
            # ordinary bounded search only at this target transition.
            self.diagnostics["committed_fallbacks"] += 1
            self.committed_target = None
            self._plan_once(observation, position, velocity, now)

        if len(self.route_events) > event_count:
            _, first, second, _, _ = self.route_events[-1]
            if self.target_index == first and second != self.committed_target:
                self.committed_target = int(second)
                self.diagnostics["commitment_updates"] += 1
