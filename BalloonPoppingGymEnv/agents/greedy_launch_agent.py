"""A deliberately simple, legal baseline for the Balloon Popping Challenge.

It uses only values supplied in ``observation`` and ``given_parameters``.  The
policy waits for a released balloon, selects the nearest one in the horizontal
plane, and aims the launch rail at that balloon.  After launch it holds neutral
TVC and full throttle.  This is a launch-and-measure baseline, not a finished
guidance controller.
"""

import numpy as np

from BalloonPoppingGymEnv.agents.base_agent import BaseAgent


class GreedyLaunchAgent(BaseAgent):
    """Aim at one balloon and stabilize the initial trajectory with rate feedback."""

    def __init__(
        self,
        given_parameters,
        min_launch_time=0.5,
        launch_inclination=70.0,
        target_lead_time=6.0,
        rate_kp=100.0,
        roll_kp=20.0,
    ):
        super().__init__(given_parameters)
        self.min_launch_time = float(min_launch_time)
        self.launch_inclination = float(launch_inclination)
        self.target_lead_time = float(target_lead_time)
        self.rate_kp = float(rate_kp)
        self.roll_kp = float(roll_kp)
        self.launched = False
        self.launch_attitude = np.array([90.0, 0.0])
        self.launch_position = np.array(
            [0.0, 0.0, float(given_parameters["environment"]["elevation"])]
        )
        control = given_parameters["rocket"]["control"]
        self.max_gimbal = float(control["max_gimbal_angle"])
        self.max_roll_torque = float(control["max_roll_torque"])

    @staticmethod
    def _choose_target(observation):
        """Return the nearest released balloon with an estimated drift velocity.

        A balloon reports almost zero velocity on its release step.  Older
        released balloons already reveal the shared wind-driven motion, so use
        their median velocity as the new balloon's first drift estimate.
        """
        states = np.asarray(observation["balloon_states"], dtype=float)
        status = np.asarray(observation["balloon_status"], dtype=int).reshape(-1)
        candidates = np.flatnonzero(status == 1)  # 1 means released
        if candidates.size == 0:
            return None

        horizontal_range_sq = np.sum(states[candidates, :2] ** 2, axis=1)
        target = states[candidates[np.argmin(horizontal_range_sq)]].copy()
        released_velocities = states[candidates, 3:6]
        moving = np.linalg.norm(released_velocities, axis=1) > 0.5
        if np.linalg.norm(target[3:6]) <= 0.5 and np.any(moving):
            target[3:6] = np.median(released_velocities[moving], axis=0)
        return target

    def _attitude_to(self, target):
        """Convert a moving ENU target into a launch inclination and heading.

        RocketPy uses x=east, y=north, z=up; heading is clockwise from north.
        A short constant-velocity lead reduces the miss caused by balloon drift.
        Inclination is kept as a tunable vehicle parameter: simply pointing at
        the balloon is too shallow for the rocket to overcome gravity.
        """
        target = np.asarray(target, dtype=float)
        predicted_position = target[:3] + self.target_lead_time * target[3:6]
        delta = predicted_position - self.launch_position
        horizontal_range = np.hypot(delta[0], delta[1])
        heading = np.degrees(np.arctan2(delta[0], delta[1])) % 360.0
        return np.array([np.clip(self.launch_inclination, 45.0, 89.0), heading])

    def _stabilizing_commands(self, observation):
        """Damp body rates using the same sign convention as the example PID."""
        sensors = np.asarray(observation["rocket_sensors"], dtype=float)
        if not self.launched or not np.all(np.isfinite(sensors[:3])):
            return np.zeros(2), 0.0

        gyro = sensors[:3]
        tvc = np.clip(-self.rate_kp * gyro[:2], -self.max_gimbal, self.max_gimbal)
        roll = float(
            np.clip(
                -self.roll_kp * gyro[2],
                -self.max_roll_torque,
                self.max_roll_torque,
            )
        )
        return tvc, roll

    def get_action(self, observation):
        target = self._choose_target(observation)
        ready_to_launch = (
            not self.launched
            and target is not None
            and observation["simulation_time"] >= self.min_launch_time
        )

        if ready_to_launch:
            self.launch_attitude = self._attitude_to(target)
            self.launched = True

        tvc, roll = self._stabilizing_commands(observation)

        return {
            "launch": self.launched,
            "launch_inclination_heading": self.launch_attitude,
            "tvc": tvc,
            "roll": roll,
            "throttle": 1.0,
        }
