"""Two-target planning to the near side of the finite balloon capture volume.

Only a private observation copy is shifted. The simulator, actual balloon
positions, and collision test are unchanged. A strict fraction below one leaves
some geometric margin for tracking/prediction error.
"""

import numpy as np

from BalloonPoppingGymEnv.agents.spline_route_agent import SplineRouteAgent


class CaptureSplineAgent(SplineRouteAgent):
    def __init__(self, given_parameters, capture_fraction=.7, **kwargs):
        super().__init__(given_parameters, **kwargs)
        self.capture_fraction = float(capture_fraction)
        if not 0. <= self.capture_fraction < 1.:
            raise ValueError('capture_fraction must be in [0, 1)')

    def _make_plan(self, observation, position, velocity, now):
        shifted = dict(observation)
        states = np.array(observation['balloon_states'], dtype=float, copy=True)
        displacement = position-states[:, :3]
        distance = np.linalg.norm(displacement, axis=1)
        offset = np.minimum(self.radius*self.capture_fraction, distance)
        states[:, :3] += displacement*(offset/np.maximum(distance, 1e-9))[:, None]
        shifted['balloon_states'] = states
        super()._make_plan(shifted, position, velocity, now)
