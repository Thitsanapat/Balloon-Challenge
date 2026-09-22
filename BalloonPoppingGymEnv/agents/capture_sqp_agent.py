"""SQP interception with a conservative aim point inside the balloon sphere."""

import numpy as np

from BalloonPoppingGymEnv.agents.optimized_spline_agent import OptimizedSplineAgent


class CaptureSQPAgent(OptimizedSplineAgent):
    def __init__(self, given_parameters, capture_fraction=0.5, **kwargs):
        super().__init__(given_parameters, **kwargs)
        self.capture_fraction = float(capture_fraction)
        if not 0 <= self.capture_fraction < 1:
            raise ValueError('capture_fraction must be in [0, 1)')

    def _make_plan(self, observation, position, velocity, now):
        # The physical radius and scoring are unchanged. Aim slightly inside
        # the near hemisphere to reduce travel, leaving a tracking-error margin.
        planned_observation = dict(observation)
        states = np.array(observation['balloon_states'], dtype=float, copy=True)
        delta = position - states[:, :3]
        distances = np.linalg.norm(delta, axis=1)
        offsets = np.minimum(self.radius * self.capture_fraction, distances)
        states[:, :3] += delta * (offsets / np.maximum(distances, 1e-9))[:, None]
        planned_observation['balloon_states'] = states
        super()._make_plan(planned_observation, position, velocity, now)
