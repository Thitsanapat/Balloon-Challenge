"""Experimental observation-only first-target ZEM guidance for Scenario 4.

The existing planner still chooses the target and spline.  Between two and
five seconds before the active target's planned arrival, this subclass adds a
bounded horizontal Zero-Effort-Miss correction to the *desired thrust vector*.
It hooks immediately before the base attitude/actuator command so those
actuators are updated exactly once per environment step.  No seed, simulator
state, or prerecorded trajectory is used.

This remains experimental until compared with fresh paired official episodes.
"""

import numpy as np

from BalloonPoppingGymEnv.agents.scenario4_wind_profile_agent import (
    Scenario4WindProfileAgent,
)
from BalloonPoppingGymEnv.agents.submission_sqp_v2 import (
    G,
    allocate_acceleration,
)


def zero_effort_miss_acceleration(relative_position, relative_velocity,
                                  time_to_go):
    """Constant acceleration needed to close the observed relative gap.

    Assumes the target keeps its currently observed velocity only over the
    short time-to-go.  This is recalculated on every sensor update; it is not
    a stored target trajectory.
    """
    relative_position = np.asarray(relative_position, dtype=float)
    relative_velocity = np.asarray(relative_velocity, dtype=float)
    time_to_go = float(time_to_go)
    if (relative_position.shape != (3,) or relative_velocity.shape != (3,) or
            not np.all(np.isfinite(relative_position)) or
            not np.all(np.isfinite(relative_velocity)) or
            not np.isfinite(time_to_go) or time_to_go <= 0):
        raise ValueError("ZEM requires finite relative state and positive time")
    return (2.0 / time_to_go**2) * (
        relative_position + relative_velocity * time_to_go)


def smooth_window_weight(time_to_go, lower, upper, fade, strength):
    """C1-continuous gate so turning ZEM on/off does not jump the command."""
    values = np.asarray((time_to_go, lower, upper, fade, strength), dtype=float)
    if (not np.all(np.isfinite(values)) or lower < 0 or upper <= lower or
            fade <= 0 or 2 * fade > upper - lower or not 0 <= strength <= 1):
        raise ValueError("Invalid ZEM time window")
    if time_to_go <= lower or time_to_go >= upper:
        return 0.0

    def smoothstep(value):
        x = float(np.clip(value, 0.0, 1.0))
        return x * x * (3.0 - 2.0 * x)

    entrance = smoothstep((time_to_go - lower) / fade)
    exit_ = smoothstep((upper - time_to_go) / fade)
    return float(strength * entrance * exit_)


def bounded_zem_force(path_force, relative_position, relative_velocity,
                      disturbance, time_to_go, weight, max_lateral_delta,
                      available, max_tilt, vertical_weight=0.0):
    """Blend bounded ZEM force with the existing already-feasible spline force.

    The existing vertical support is retained by default.  Final allocation
    uses the same public thrust/tilt envelope as the base controller.
    """
    path_force = np.asarray(path_force, dtype=float)
    disturbance = np.asarray(disturbance, dtype=float)
    scalars = (time_to_go, weight, max_lateral_delta, available,
               max_tilt, vertical_weight)
    if (path_force.shape != (3,) or disturbance.shape != (3,) or
            not np.all(np.isfinite(path_force)) or
            not np.all(np.isfinite(disturbance)) or
            not np.all(np.isfinite(scalars)) or time_to_go <= 0 or
            not 0 <= weight <= 1 or max_lateral_delta <= 0 or available < 0 or
            not 0 <= max_tilt < np.pi / 2 or not 0 <= vertical_weight <= 1):
        raise ValueError("Invalid bounded-ZEM input")
    zem_acceleration = zero_effort_miss_acceleration(
        relative_position, relative_velocity, time_to_go)
    zem_force = zem_acceleration - G - disturbance
    correction = zem_force[:2] - path_force[:2]
    length = float(np.linalg.norm(correction))
    if length > max_lateral_delta:
        correction *= max_lateral_delta / length
    requested = path_force.copy()
    requested[:2] += weight * correction
    requested[2] += weight * vertical_weight * (
        zem_force[2] - path_force[2])
    allocated = allocate_acceleration(requested, available, max_tilt)
    saturated = float(np.linalg.norm(allocated - requested)) > 0.1
    return allocated, saturated


def _slew_axis(previous, candidate, max_angle):
    previous = np.asarray(previous, dtype=float)
    candidate = np.asarray(candidate, dtype=float)
    angle = float(np.arccos(np.clip(np.dot(previous, candidate), -1.0, 1.0)))
    if angle <= max_angle or angle <= 1e-12:
        return candidate.copy(), False
    # Normalized linear interpolation is stable for the small per-step angles
    # used here and keeps the axis inside the convex thrust-tilt cone.
    fraction = max_angle / angle
    direction = (1.0 - fraction) * previous + fraction * candidate
    direction /= np.linalg.norm(direction)
    return direction, True


class Scenario4ZemTerminalAgent(Scenario4WindProfileAgent):
    """Add bounded relative-state homing without double actuator updates."""

    def __init__(self, given_parameters, zem_start_seconds=5.0,
                 zem_stop_seconds=2.0, zem_fade_seconds=0.4,
                 zem_strength=0.6, zem_max_lateral_delta=5.0,
                 zem_max_target_range=150.0, zem_vertical_weight=0.0,
                 zem_axis_slew_rate=None, **kwargs):
        super().__init__(given_parameters, **kwargs)
        self.zem_start_seconds = float(zem_start_seconds)
        self.zem_stop_seconds = float(zem_stop_seconds)
        self.zem_fade_seconds = float(zem_fade_seconds)
        self.zem_strength = float(zem_strength)
        self.zem_max_lateral_delta = float(zem_max_lateral_delta)
        self.zem_max_target_range = float(zem_max_target_range)
        self.zem_vertical_weight = float(zem_vertical_weight)
        self.zem_axis_slew_rate = float(
            self.max_axis_rate if zem_axis_slew_rate is None else
            zem_axis_slew_rate)
        # Validate all settings, including the geometry of the smooth window.
        smooth_window_weight(
            0.5 * (self.zem_start_seconds + self.zem_stop_seconds),
            self.zem_stop_seconds, self.zem_start_seconds,
            self.zem_fade_seconds, self.zem_strength)
        values = (self.zem_max_lateral_delta, self.zem_max_target_range,
                  self.zem_vertical_weight, self.zem_axis_slew_rate)
        if (not np.all(np.isfinite(values)) or
                min(values[:2]) <= 0 or not 0 <= self.zem_vertical_weight <= 1 or
                self.zem_axis_slew_rate <= 0):
            raise ValueError("Invalid ZEM correction limits")
        self._zem_snapshot = None
        self._zem_previous_axis = None
        self.diagnostics.update(zem_applied_steps=0,
                                zem_saturated_steps=0,
                                zem_slew_limited_steps=0)

    def get_action(self, observation):
        # Snapshot only released-status and currently observed balloon state.
        # The planner may choose a target during this same get_action call.
        self._zem_snapshot = (
            float(observation["simulation_time"]),
            np.asarray(observation["balloon_status"]).reshape(-1).copy(),
            np.asarray(observation["balloon_states"], dtype=float).copy(),
        )
        try:
            return super().get_action(observation)
        finally:
            self._zem_snapshot = None

    def _attitude_action(self, axis, jerk, magnitude, gyro, available):
        baseline_axis = np.asarray(axis, dtype=float)
        final_axis = baseline_axis
        final_magnitude = float(magnitude)
        final_jerk = jerk
        snapshot = self._zem_snapshot
        target = self.target_index
        if (snapshot is not None and target is not None and
                self.filtered_position is not None and
                self.filtered_velocity is not None):
            now, status, states = snapshot
            if (0 <= target < len(status) and status[target] == 1 and
                    np.all(np.isfinite(states[target])) and
                    np.all(np.isfinite(self.filtered_position)) and
                    np.all(np.isfinite(self.filtered_velocity))):
                deadline = (self.deadlines[0]
                            if self.route and self.deadlines and
                            self.route[0] == target else
                            self.plan_start + self.plan_duration)
                time_to_go = float(deadline - now)
                weight = smooth_window_weight(
                    time_to_go, self.zem_stop_seconds,
                    self.zem_start_seconds, self.zem_fade_seconds,
                    self.zem_strength)
                relative_position = states[target, :3] - self.filtered_position
                if (weight > 0 and
                        np.linalg.norm(relative_position) <=
                        self.zem_max_target_range):
                    relative_velocity = (
                        states[target, 3:6] - self.filtered_velocity)
                    force, saturated = bounded_zem_force(
                        baseline_axis * final_magnitude,
                        relative_position, relative_velocity,
                        self.disturbance, time_to_go, weight,
                        self.zem_max_lateral_delta, available, self.max_tilt,
                        self.zem_vertical_weight)
                    candidate_magnitude = float(np.linalg.norm(force))
                    if (np.all(np.isfinite(force)) and
                            candidate_magnitude > 1e-9):
                        candidate_axis = force / candidate_magnitude
                        if self._zem_previous_axis is not None:
                            candidate_axis, slew_limited = _slew_axis(
                                self._zem_previous_axis, candidate_axis,
                                self.zem_axis_slew_rate * self.dt)
                            self.diagnostics["zem_slew_limited_steps"] += int(
                                slew_limited)
                        final_axis = candidate_axis
                        final_magnitude = candidate_magnitude
                        # The spline jerk is no longer the derivative of the
                        # mixed force, so its attitude-rate feedforward is not
                        # valid during the ZEM window.
                        final_jerk = np.zeros(3)
                        self.diagnostics["zem_applied_steps"] += 1
                        self.diagnostics["zem_saturated_steps"] += int(saturated)
        self._zem_previous_axis = np.asarray(final_axis, dtype=float).copy()
        self.last_desired_axis = self._zem_previous_axis.copy()
        return super()._attitude_action(
            final_axis, final_jerk, final_magnitude, gyro, available)
