"""Experimental expanding-spiral waypoints in an observed balloon-drift frame.

The frame axis is not the rocket's thrust axis. Actual attitude/force constraints
are enforced by PhysicsGuidanceAgent. No hidden wind or future paths are read.
"""

import numpy as np

from BalloonPoppingGymEnv.agents.physics_guidance_agent import (
    G, PhysicsGuidanceAgent, quintic_intercept,
)


def drift_frame(drift, max_tilt):
    """Right-handed frame tilted downwind, with angle inferred from ascent/drift."""
    drift = np.asarray(drift, dtype=float)
    horizontal = np.linalg.norm(drift[:2])
    heading = drift[:2] / horizontal if horizontal > 1e-8 else np.array([1., 0.])
    tilt = min(max_tilt, np.arctan2(horizontal, max(drift[2], 0.1)))
    axis = np.r_[np.sin(tilt)*heading, np.cos(tilt)]
    crosswind = np.array([-heading[1], heading[0], 0.])
    radial = np.cross(crosswind, axis)
    return axis, radial, crosswind


def spiral_state(center, drift, radial, crosswind, radius, growth, phase, omega, dt):
    """Position/velocity/acceleration of a linearly expanding, translating spiral."""
    angle = phase + omega*dt
    r = radius + growth*dt
    u = np.cos(angle)*radial + np.sin(angle)*crosswind
    tangent = -np.sin(angle)*radial + np.cos(angle)*crosswind
    p = center + drift*dt + r*u
    v = drift + growth*u + r*omega*tangent
    a = 2*growth*omega*tangent - r*omega**2*u
    return p, v, a


class WindSpiralAgent(PhysicsGuidanceAgent):
    def __init__(self, given_parameters, spiral_axis_tilt=45., initial_radius=2.,
                 radius_growth=0.35, angular_speed=0.4, start_after_first_pop=True, **kwargs):
        super().__init__(given_parameters, **kwargs)
        if not (0 <= spiral_axis_tilt < 90 and initial_radius > 0
                and radius_growth >= 0 and angular_speed > 0):
            raise ValueError("Invalid spiral geometry")
        self.spiral_axis_tilt = np.radians(spiral_axis_tilt)
        self.initial_radius = float(initial_radius)
        self.radius_growth = float(radius_growth)
        self.angular_speed = float(angular_speed)
        self.start_after_first_pop = bool(start_after_first_pop)
        self.filtered_drift = None
        self.frame_time = None
        self.spiral_start = None
        self.spiral_center = None
        self.phase = 0.
        self.diagnostics.update(spiral_frames=0, axis_tilt_degrees=0.,
                                spiral_radius=initial_radius, drift=[0., 0., 0.])

    def _spiral_geometry(self, observation, position, now):
        states = np.asarray(observation["balloon_states"], dtype=float)
        status = np.asarray(observation["balloon_status"]).reshape(-1)
        released = status == 1
        moving = released & (np.linalg.norm(states[:, 3:6], axis=1) > 0.5)
        # Enter the field through a feasible intercept first. A global cloud
        # centroid can be unreachable from the pad within the capture horizon.
        if not np.any(moving) or (self.start_after_first_pop and self.spiral_start is None
                                  and not np.any(status == 2)):
            return None
        observed = np.median(states[moving, 3:6], axis=0)
        elapsed = 0. if self.frame_time is None else max(0., now-self.frame_time)
        if self.filtered_drift is None:
            self.filtered_drift = observed.copy()
        else:
            self.filtered_drift += elapsed/(1.+elapsed)*(observed-self.filtered_drift)
        self.frame_time = now
        axis, radial, crosswind = drift_frame(self.filtered_drift, self.spiral_axis_tilt)
        age = max(0., now-self.launch_time)
        if self.spiral_start is None:
            self.spiral_start = now
            self.spiral_center = position-self.initial_radius*radial
        else:
            self.spiral_center += self.filtered_drift*elapsed
        radius = self.initial_radius + self.radius_growth*(now-self.spiral_start)
        remaining = max(0., self.burn_time-age)
        largest_radius = radius + self.radius_growth*remaining
        reserve = np.sqrt(max(0., self.available_acceleration(now)**2-G[2]**2))
        # Reserve most lateral authority for capture/tracking; full curves checked below.
        omega = min(self.angular_speed, np.sqrt(0.25*reserve/max(largest_radius, 1e-9)))
        self.phase += omega*elapsed
        center = self.spiral_center
        self.diagnostics.update(spiral_frames=self.diagnostics["spiral_frames"]+1,
                                axis_tilt_degrees=float(np.degrees(np.arccos(axis[2]))),
                                spiral_radius=float(radius), drift=self.filtered_drift.tolist())
        return center, self.filtered_drift, radial, crosswind, radius, self.radius_growth, self.phase, omega

    def _make_plan(self, observation, position, velocity, now):
        geometry = self._spiral_geometry(observation, position, now)
        if geometry is None:
            super()._make_plan(observation, position, velocity, now)
            return
        remaining = max(0., self.burn_time-(now-self.launch_time))
        self.target_index = None
        for duration in np.arange(0.8, min(self.horizon, remaining)+0.01, 0.4):
            target, terminal_v, terminal_a = spiral_state(*geometry, duration)
            curve = quintic_intercept(position, velocity, self.acceleration,
                                      target, terminal_v, duration)
            # Complete the quintic boundary conditions with nonzero spiral acceleration.
            correction = duration**2*terminal_a
            curve[3] += correction/2
            curve[4] -= correction
            curve[5] += correction/2
            if self._feasible(curve, duration, now):
                self.plan, self.plan_start, self.plan_duration = curve, now, duration
                self.diagnostics["plans"] += 1
                return
        self.diagnostics["no_feasible_plan"] += 1
        if self.plan is not None and now >= self.plan_start+self.plan_duration:
            self.plan = None
