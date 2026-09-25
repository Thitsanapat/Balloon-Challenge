"""Complete a live final intercept instead of abandoning it just before arrival.

The original planner's 0.12-second short-horizon cutoff can drop a target just
outside its capture sphere. Retain the existing reference until its deadline,
then review immediately if no hit was observed. No horizon extension or target
position knowledge beyond observations is used.
"""

import numpy as np
from BalloonPoppingGymEnv.agents.chain_launch_agent import ChainLaunchAgent
from BalloonPoppingGymEnv.agents.constrained_beam_agent import ConstrainedBeamAgent


class FinalApproachMixin:
    def __init__(self, given_parameters, final_approach_guard=.15, **kwargs):
        super().__init__(given_parameters, **kwargs)
        self.final_approach_guard = float(final_approach_guard)
        if not np.isfinite(self.final_approach_guard) or self.final_approach_guard < 0:
            raise ValueError('Nonnegative finite final approach guard required')
        self.diagnostics['final_approach_holds'] = 0

    def _make_plan(self, observation, position, velocity, now):
        if self.plan is not None and self.target_index is not None:
            remaining = self.plan_start+self.plan_duration-now
            status = np.asarray(observation['balloon_status']).reshape(-1)
            if 0 < remaining <= self.final_approach_guard and status[self.target_index] == 1:
                self.diagnostics['final_approach_holds'] += 1
                return
        return super()._make_plan(observation, position, velocity, now)

    def get_action(self, observation):
        action = super().get_action(observation)
        if self.plan is not None and self.target_index is not None:
            now = float(observation['simulation_time'])
            deadline = self.plan_start+self.plan_duration
            if 0 < deadline-now <= self.final_approach_guard:
                # A missed target must not defer the next decision by the full
                # replan interval after the old reference has already expired.
                self.next_plan = min(self.next_plan, deadline)
        return action


class FinalApproachAgent(FinalApproachMixin, ChainLaunchAgent):
    pass


class FinalApproachConstrainedAgent(FinalApproachMixin, ConstrainedBeamAgent):
    pass
