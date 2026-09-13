"""Diagnostic spline agent that forces an observed target order.

This is for feasibility experiments on recorded fields, not the random-seed
competition policy.  All target states still come from the live observation.
"""

import numpy as np

from BalloonPoppingGymEnv.agents.spline_route_agent import SplineRouteAgent


class ScriptedSplineAgent(SplineRouteAgent):
    def __init__(self, given_parameters, scripted_targets, **kwargs):
        super().__init__(given_parameters, **kwargs)
        self.scripted_targets = [int(index) for index in scripted_targets]
        self.scripted_pointer = 0
        self.diagnostics.update(scripted_attempts=0)

    def _next_scripted_target(self, observation):
        status = np.asarray(observation["balloon_status"]).reshape(-1)
        while self.scripted_pointer < len(self.scripted_targets):
            index = self.scripted_targets[self.scripted_pointer]
            if status[index] == 2:
                self.scripted_pointer += 1
                continue
            return index if status[index] == 1 else None
        return None

    def _make_plan(self, observation, position, velocity, now):
        desired = self._next_scripted_target(observation)
        if desired is None:
            super()._make_plan(observation, position, velocity, now)
            return
        self.diagnostics["scripted_attempts"] += 1
        released = np.flatnonzero(
            np.asarray(observation["balloon_status"]).reshape(-1) == 1
        )
        saved = self.failed_until.copy()
        try:
            for index in released:
                if int(index) != desired:
                    self.failed_until[int(index)] = np.inf
            super()._make_plan(observation, position, velocity, now)
        finally:
            self.failed_until = saved
