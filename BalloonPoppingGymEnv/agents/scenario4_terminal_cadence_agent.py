"""Experimental observation-paced replanning on the active intercept.

The official joint-route planner is unchanged. During the last few seconds
before a committed intercept, refresh the reference at half its normal
interval so it sees new target and rocket measurements earlier. This adds no
hidden-state access, precomputed trajectory, or extra control authority.
"""

import numpy as np

from BalloonPoppingGymEnv.agents.scenario4_pure_joint_refresh_agent import (
    Scenario4PureJointRefreshAgent,
)


class Scenario4TerminalCadenceAgent(Scenario4PureJointRefreshAgent):
    def __init__(self, given_parameters, terminal_replan_interval=0.2,
                 terminal_replan_start=3.0, terminal_replan_stop=0.6,
                 **kwargs):
        super().__init__(given_parameters, **kwargs)
        self.terminal_replan_interval = float(terminal_replan_interval)
        self.terminal_replan_start = float(terminal_replan_start)
        self.terminal_replan_stop = float(terminal_replan_stop)
        if (not np.all(np.isfinite((self.terminal_replan_interval,
                                    self.terminal_replan_start,
                                    self.terminal_replan_stop))) or
                not 0 < self.terminal_replan_interval < self.replan_interval or
                not 0 < self.terminal_replan_stop < self.terminal_replan_start):
            raise ValueError("Invalid terminal replanning window")
        self.diagnostics["terminal_cadence_steps"] = 0

    def get_action(self, observation):
        normal = self.replan_interval
        now = float(observation["simulation_time"])
        status = np.asarray(observation["balloon_status"]).reshape(-1)
        target = self.target_index
        if (self.plan is not None and target is not None and
                0 <= target < len(status) and status[target] == 1):
            remaining = self.plan_start + self.plan_duration - now
            if self.terminal_replan_stop <= remaining <= self.terminal_replan_start:
                self.replan_interval = self.terminal_replan_interval
                self.next_plan = min(self.next_plan,
                                     now + self.terminal_replan_interval)
                self.diagnostics["terminal_cadence_steps"] += 1
        try:
            return super().get_action(observation)
        finally:
            self.replan_interval = normal
