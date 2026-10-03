"""Choose a launch direction whose first interception can absorb forecast error.

The test uses the first planned curve and the published rocket constraints.
At two points along that curve, it rebuilds the remaining leg to nearby
horizontal target positions and checks each correction densely.  These are
counterfactual *planning* targets, not hidden future balloon states.  The
policy sees only the current observation and ``given_parameters``.
"""

import numpy as np

from BalloonPoppingGymEnv.agents.scenario4_wind_profile_agent import (
    Scenario4WindProfileAgent,
)
from BalloonPoppingGymEnv.agents.submission_sqp_v2 import (
    boundary_curve, sample_curve,
)


class Scenario4RecoverableLaunchAgent(Scenario4WindProfileAgent):
    """Rank launch candidates by plausible first-leg correction capacity.

    ``correction_radii`` are horizontal metres of endpoint error to test.
    ``correction_fractions`` are times in the first leg at which an updated
    balloon observation could prompt a correction.  The search still chooses
    only trajectories that pass the ordinary dense feasibility check.
    """

    def __init__(self, given_parameters, correction_radii=(1.5, 3.0, 5.0),
                 correction_fractions=(0.5, 0.75),
                 continuation_credit=0.12, robustness_floor=0.05,
                 first_time_cost=0.08, **kwargs):
        super().__init__(given_parameters, **kwargs)
        self.correction_radii = np.asarray(correction_radii, dtype=float)
        self.correction_fractions = np.asarray(correction_fractions, dtype=float)
        self.continuation_credit = float(continuation_credit)
        self.robustness_floor = float(robustness_floor)
        self.first_time_cost = float(first_time_cost)
        if (self.correction_radii.ndim != 1 or not len(self.correction_radii)
                or np.any(~np.isfinite(self.correction_radii))
                or np.any(self.correction_radii <= 0)
                or self.correction_fractions.ndim != 1
                or not len(self.correction_fractions)
                or np.any(~np.isfinite(self.correction_fractions))
                or np.any((self.correction_fractions <= 0) |
                           (self.correction_fractions >= 1))
                or not np.all(np.isfinite((self.continuation_credit,
                                          self.robustness_floor,
                                          self.first_time_cost)))
                or self.continuation_credit < 0
                or not 0 <= self.robustness_floor <= 1
                or self.first_time_cost < 0):
            raise ValueError("Invalid recoverable-launch settings")

    def _correction_fraction(self, duration, now):
        """Return fraction of feasible position corrections from two replans.

        All headings are checked symmetrically so one favorable wind direction
        cannot make a launch axis look robust.  The test is a reachability
        proxy; live guidance still replans from each observed state.
        """
        directions = np.array(((1., 0., 0.), (-1., 0., 0.),
                               (0., 1., 0.), (0., -1., 0.),
                               (1., 1., 0.), (1., -1., 0.),
                               (-1., 1., 0.), (-1., -1., 0.)))
        directions[4:] /= np.sqrt(2.)
        terminal_p, terminal_v, terminal_a, _ = sample_curve(
            self.plan, duration, duration)
        total = feasible = 0
        for fraction in self.correction_fractions:
            elapsed = duration * fraction
            remaining = duration - elapsed
            start_p, start_v, start_a, _ = sample_curve(
                self.plan, duration, elapsed)
            curves = np.asarray([
                boundary_curve(start_p, start_v, start_a,
                               terminal_p + radius * direction,
                               terminal_v, terminal_a, remaining)
                for radius in self.correction_radii
                for direction in directions
            ])
            valid = self._valid(
                curves, np.full(len(curves), remaining),
                np.full((len(curves), 1), now + elapsed), samples=65)[0]
            total += len(valid)
            feasible += int(np.count_nonzero(valid))
        return feasible / total

    def _launch_candidate_key(self, axis, count, finish, observation, now):
        if (not count or self.plan is None or not self.route or
                not self.deadlines):
            return (1, float('inf'), float('inf'), 0, float('inf'))
        duration = float(self.deadlines[0] - now)
        if not np.isfinite(duration) or duration <= 0:
            return (1, float('inf'), float('inf'), 0, float('inf'))
        robust = self._correction_fraction(duration, now)
        # The first hit earns full credit; later nominal hits have diminishing
        # credit because each depends on future target-motion predictions.
        route_value = 1. + self.continuation_credit * (count - 1)
        confidence = self.robustness_floor + (1. - self.robustness_floor) * robust
        value = route_value * confidence - self.first_time_cost * duration
        return (0, -value, duration, -count, finish)
