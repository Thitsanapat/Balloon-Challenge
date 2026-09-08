"""Spline guidance with an observation-selected initial launch attitude.

The launch attitude is part of the six-degree-of-freedom initial condition.
Choosing it before ignition avoids spending the first seconds and finite TVC
authority rotating away from an upright launch.
"""

import numpy as np

from BalloonPoppingGymEnv.agents.multi_target_agent import _rotate_body_to_world
from BalloonPoppingGymEnv.agents.spline_route_agent import SplineRouteAgent
from BalloonPoppingGymEnv.envs.balloon_world import get_initial_attitude


class InclinedSplineAgent(SplineRouteAgent):
    def __init__(self, given_parameters, launch_inclination=70.0,
                 launch_lead_time=4.0, initial_target_index=None,
                 initial_cluster_weight=0.0, **kwargs):
        super().__init__(given_parameters, **kwargs)
        self.launch_inclination = float(launch_inclination)
        self.launch_lead_time = float(launch_lead_time)
        self.initial_target_index = (
            None if initial_target_index is None else int(initial_target_index)
        )
        self.initial_cluster_weight = float(initial_cluster_weight)
        if not 0.0 <= self.launch_inclination <= 90.0:
            raise ValueError("launch_inclination must be between 0 and 90 degrees")
        if self.launch_lead_time < 0.0 or self.initial_cluster_weight < 0.0:
            raise ValueError("launch lead time and cluster weight must be nonnegative")
        self.initial_aim_index = None
        self.diagnostics.update(initial_aim_index=None, launch_attitude=[90.0, 0.0])

    def _initial_aim(self, observation):
        states = np.asarray(observation["balloon_states"], dtype=float)
        status = np.asarray(observation["balloon_status"]).reshape(-1)
        released = np.flatnonzero(status == 1)
        if released.size == 0:
            return None, None

        moving = released[np.linalg.norm(states[released, 3:6], axis=1) > 0.5]
        shared = np.median(states[moving, 3:6], axis=0) if moving.size else np.zeros(3)
        drift = states[released, 3:6].copy()
        drift[np.linalg.norm(drift, axis=1) <= 0.5] = shared
        predicted = states[released, :3] + self.launch_lead_time * drift
        pad = np.array([0.0, 0.0, self.elevation])

        if self.initial_target_index is not None and self.initial_target_index in released:
            local = int(np.flatnonzero(released == self.initial_target_index)[0])
        else:
            ranges = np.linalg.norm(predicted - pad, axis=1)
            if released.size > 1 and self.initial_cluster_weight > 0.0:
                pairwise = np.linalg.norm(
                    predicted[:, None, :] - predicted[None, :, :], axis=2
                )
                np.fill_diagonal(pairwise, np.inf)
                ranges += self.initial_cluster_weight * np.min(pairwise, axis=1)
            local = int(np.argmin(ranges))
        return int(released[local]), predicted[local]

    def _set_initial_attitude(self, observation):
        index, predicted = self._initial_aim(observation)
        if index is None:
            return
        delta = predicted - np.array([0.0, 0.0, self.elevation])
        heading = float(np.degrees(np.arctan2(delta[0], delta[1])) % 360.0)
        self.launch_attitude = np.array([self.launch_inclination, heading])
        self.quaternion = np.asarray(get_initial_attitude(*self.launch_attitude), dtype=float)
        self.last_desired_axis = _rotate_body_to_world(
            self.quaternion, np.array([0.0, 0.0, 1.0])
        )
        self.target_index = index
        self.initial_aim_index = index
        self.target_events.append((float(observation["simulation_time"]), index))
        self.diagnostics["initial_aim_index"] = index
        self.diagnostics["launch_attitude"] = self.launch_attitude.tolist()

    def get_action(self, observation):
        now = float(observation["simulation_time"])
        if not self.launched and now >= self.launch_time:
            self._set_initial_attitude(observation)
        return super().get_action(observation)
