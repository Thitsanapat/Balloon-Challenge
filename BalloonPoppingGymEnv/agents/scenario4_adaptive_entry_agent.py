"""Experimental observation-only gate for an overlong launch route.

At the launch decision, evaluate the incumbent's normal route comparison on
a copied internal state.  If its longest nominal chain has at least nine
targets in the public 30-second burn, use the shorter entry-horizon search;
otherwise keep the incumbent.  Only the chosen policy state is retained.
This is a heuristic capacity check, not a calibrated capture model.
"""

import copy

from BalloonPoppingGymEnv.agents.scenario4_entry_horizon_agent import (
    Scenario4EntryHorizonAgent,
)
from BalloonPoppingGymEnv.agents.submission_scenario4_pure_joint_v2 import (
    Scenario4PureJointRefreshAgent,
)


class Scenario4AdaptiveEntryAgent(Scenario4EntryHorizonAgent):
    def __init__(self, given_parameters, overlong_route_count=9, **kwargs):
        super().__init__(given_parameters, **kwargs)
        self.overlong_route_count = int(overlong_route_count)
        if self.overlong_route_count < 2:
            raise ValueError("overlong_route_count must be at least two")
        self.adaptive_decided = False
        self.diagnostics.update(adaptive_full_nominal_count=None,
                                adaptive_short_entry=None)

    def get_action(self, observation):
        now = float(observation['simulation_time'])
        if (not self.adaptive_decided and not self.launched and
                not self.launch_selected and now >= self.launch_time):
            original = copy.deepcopy(self.__dict__)
            try:
                # A planning-only probe. The returned action is discarded and
                # the complete agent state is restored before the real choice.
                Scenario4PureJointRefreshAgent.get_action(self, observation)
                count = max((int(item[1]) for item in self.launch_comparisons),
                            default=0)
            finally:
                self.__dict__ = original
            self.entry_pending = count >= self.overlong_route_count
            self.adaptive_decided = True
            self.diagnostics['adaptive_full_nominal_count'] = count
            self.diagnostics['adaptive_short_entry'] = self.entry_pending
        return super().get_action(observation)
