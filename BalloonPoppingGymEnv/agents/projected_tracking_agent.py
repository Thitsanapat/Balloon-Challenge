"""Euclidean thrust allocation for observation-only reference tracking.

Project requested force onto the tilt cone intersected with the available
thrust ball. Unlike vertical-priority clipping, this minimizes total squared
acceleration error under saturation. It does not increase available thrust.
"""

import numpy as np
from BalloonPoppingGymEnv.agents.final_approach_agent import (
    FinalApproachAgent, FinalApproachConstrainedAgent,
)
from BalloonPoppingGymEnv.agents.submission_sqp_v2 import sample_curve, G


def project_thrust(requested, available, max_tilt):
    force = np.asarray(requested, dtype=float).copy()
    available = max(float(available), 0.)
    radial = float(np.linalg.norm(force[:2]))
    sine, cosine = np.sin(max_tilt), np.cos(max_tilt)
    if force[2] < 0 or radial*cosine > force[2]*sine:
        length = max(0., radial*sine+force[2]*cosine)
        force[:2] *= length*sine/max(radial, 1e-12)
        force[2] = length*cosine
    norm = float(np.linalg.norm(force))
    if norm > available:
        force *= available/max(norm, 1e-12)
    return force


class ProjectedTrackingMixin:
    def get_action(self, observation):
        self.control_time = float(observation['simulation_time'])
        return super().get_action(observation)

    def _attitude_action(self, axis, jerk, magnitude, gyro, available):
        if self.plan is not None and available > 0:
            elapsed = np.clip(self.control_time-self.plan_start, 0., self.plan_duration)
            p, v, a, jerk = sample_curve(self.plan, self.plan_duration, elapsed)
            wn = self.tracking_frequency
            requested = a+wn**2*(p-self.filtered_position)+2*wn*(v-self.filtered_velocity)-G-self.disturbance
            if self.minimum_vertical_acceleration is not None:
                requested[2] = max(requested[2], -G[2]+self.minimum_vertical_acceleration)
            force = project_thrust(requested, available, self.max_tilt)
            magnitude = float(np.linalg.norm(force))
            axis = force/max(magnitude, 1e-9) if magnitude > 1e-9 else np.array([0., 0., 1.])
            self.last_desired_axis = axis.copy()
        return super()._attitude_action(axis, jerk, magnitude, gyro, available)


class ProjectedTrackingAgent(ProjectedTrackingMixin, FinalApproachAgent):
    pass


class ProjectedConstrainedAgent(ProjectedTrackingMixin, FinalApproachConstrainedAgent):
    pass
