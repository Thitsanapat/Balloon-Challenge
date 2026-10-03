"""Choose the launch rail direction for a reachable first balloon.

The ordinary launch comparison maximizes the length of a nominal route. In
Scenario 4, that route often has an eight-second first leg whose moving-target
forecast can drift by metres. This isolated experiment favors an earlier first
intercept with room to correct its trajectory after launch. It uses only the
candidate plan, current observation, and public vehicle parameters.
"""

import numpy as np

from BalloonPoppingGymEnv.agents.scenario4_wind_profile_agent import (
    Scenario4WindProfileAgent,
)
from BalloonPoppingGymEnv.agents.submission_sqp_v2 import G, sample_curve


class Scenario4FirstHitLaunchAgent(Scenario4WindProfileAgent):
    """Rank launch axes by first-leg duration and lateral thrust headroom.

    ``launch_margin_weight`` controls how strongly small acceleration headroom
    is penalized, in seconds times m/s². ``launch_margin_floor`` prevents a
    singular score near full thrust. ``launch_beyond_profile_weight`` adds a
    penalty per second beyond the wind forecaster's supported horizon. Route
    length and total time break ties. Once airborne, the established planner
    and controller are unchanged.
    """

    def __init__(self, given_parameters, launch_margin_weight=1.0,
                 launch_margin_floor=0.25,
                 launch_beyond_profile_weight=2.0, **kwargs):
        super().__init__(given_parameters, **kwargs)
        self.launch_margin_weight = float(launch_margin_weight)
        self.launch_margin_floor = float(launch_margin_floor)
        self.launch_beyond_profile_weight = float(launch_beyond_profile_weight)
        values = (self.launch_margin_weight, self.launch_margin_floor,
                  self.launch_beyond_profile_weight)
        if (not np.all(np.isfinite(values)) or self.launch_margin_weight < 0 or
                self.launch_margin_floor <= 0 or
                self.launch_beyond_profile_weight < 0):
            raise ValueError("Invalid first-hit launch ranking settings")

    def _first_leg_headroom(self, duration, now):
        """Conservative late-leg lateral acceleration margin, in m/s².

        The first 50% of a leg can recover from a large early forecast error.
        The last 50% determines how much late wind error can still be corrected.
        The lower quintile avoids treating a single discretization sample as
        representative of the entire correction window.
        """
        times = np.linspace(0.5 * duration, duration, 25)
        _, _, acceleration, _ = sample_curve(self.plan, duration, times)
        force = acceleration - G - self.disturbance
        available = np.asarray([
            self.available_acceleration(now + elapsed) for elapsed in times
        ])
        lateral_limit = np.sqrt(np.maximum(available**2 - force[:, 2]**2, 0.0))
        lateral_margin = lateral_limit - np.linalg.norm(force[:, :2], axis=1)
        total_margin = available - np.linalg.norm(force, axis=1)
        margin = np.minimum(lateral_margin, total_margin)
        if not np.all(np.isfinite(margin)):
            return 0.0
        return max(0.0, float(np.percentile(margin, 20)))

    def _launch_candidate_key(self, axis, count, finish, observation, now):
        if not count or self.plan is None or not self.route or not self.deadlines:
            return (1, float("inf"), 0, float("inf"))
        duration = float(self.deadlines[0] - now)
        if not np.isfinite(duration) or duration <= 0:
            return (1, float("inf"), 0, float("inf"))
        margin = self._first_leg_headroom(duration, now)
        cost = (duration
                + self.launch_margin_weight /
                max(margin, self.launch_margin_floor)
                + self.launch_beyond_profile_weight *
                max(0.0, duration - self.profile_horizon))
        return (0, cost, -count, finish)
