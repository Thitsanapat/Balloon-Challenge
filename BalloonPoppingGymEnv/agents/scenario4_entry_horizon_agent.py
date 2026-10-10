"""Experimental short entry search with incumbent full-depth later replans.

The alternate depth applies only to the launch-direction comparison and the
first executable route.  Once the launch action has been computed, the
incumbent search depth is restored.  No seed, simulator truth, or future
balloon trajectory enters the decision.
"""

from BalloonPoppingGymEnv.agents.submission_scenario4_pure_joint_v2 import (
    Scenario4PureJointRefreshAgent,
)


class Scenario4EntryHorizonAgent(Scenario4PureJointRefreshAgent):
    def __init__(self, given_parameters, entry_search_depth=6, **kwargs):
        super().__init__(given_parameters, **kwargs)
        self.entry_search_depth = int(entry_search_depth)
        if self.entry_search_depth < 1:
            raise ValueError("entry_search_depth must be positive")
        self.entry_pending = True
        self.diagnostics.update(entry_horizon_applied=0, entry_route=[])

    def get_action(self, observation):
        now = float(observation['simulation_time'])
        if (not self.entry_pending or now < self.launch_time or
                self.entry_search_depth >= self.search_depth):
            return super().get_action(observation)
        old_depth = self.search_depth
        try:
            self.search_depth = self.entry_search_depth
            result = super().get_action(observation)
            if self.route:
                self.entry_pending = False
                self.diagnostics['entry_horizon_applied'] += 1
                self.diagnostics['entry_route'] = list(self.route)
            return result
        finally:
            self.search_depth = old_depth
