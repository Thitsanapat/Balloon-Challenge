"""Refresh shared arrival derivatives when an observed endpoint has shifted.

The same observed route and deadlines remain in place. A feasible ordinary
refresh is preserved as fallback if the full-route re-solve is infeasible.
No seed, simulator truth or prerecorded trajectories enter this policy.
"""

import numpy as np

from BalloonPoppingGymEnv.agents.scenario4_pure_joint_refresh_agent import (
    Scenario4PureJointRefreshAgent,
)


class Scenario4ProactiveJointAgent(Scenario4PureJointRefreshAgent):
    def __init__(self, given_parameters, proactive_radius_fraction=0.5,
                 proactive_min_seconds=1.0, proactive_max_seconds=6.0, **kwargs):
        super().__init__(given_parameters, **kwargs)
        values = (proactive_radius_fraction, proactive_min_seconds, proactive_max_seconds)
        if (not np.all(np.isfinite(values)) or proactive_radius_fraction <= 0 or
                proactive_min_seconds <= 0.12 or
                proactive_max_seconds <= proactive_min_seconds):
            raise ValueError('Invalid proactive refresh limits')
        self.proactive_threshold = float(proactive_radius_fraction) * self.radius
        self.proactive_min_seconds = float(proactive_min_seconds)
        self.proactive_max_seconds = float(proactive_max_seconds)
        self._pending_feasible_refresh = None
        self.diagnostics.update(proactive_requests=0, proactive_accepted=0,
                                proactive_fallbacks=0)

    def _accept_committed_curve(self, curve, duration, now):
        self._pending_feasible_refresh = None
        accepted = super()._accept_committed_curve(curve, duration, now)
        if (not accepted or self.plan is None or len(self.route) < 2 or
                self.target_index != self.route[0] or
                not self.proactive_min_seconds <= duration <= self.proactive_max_seconds):
            return accepted
        # Curves use normalized time; the endpoint at t/T=1 is the sum of
        # coefficients. The new curve includes the current observed wind fit.
        shift = float(np.linalg.norm(np.sum(curve, axis=0) - np.sum(self.plan, axis=0)))
        if not np.isfinite(shift) or shift < self.proactive_threshold:
            return accepted
        self._pending_feasible_refresh = (np.asarray(curve).copy(), duration, now)
        self.diagnostics['proactive_requests'] += 1
        return False  # Ask the existing planner to try its joint recovery hook.

    def _rescue_committed_leg(self, observation, position, velocity, now, duration):
        pending = self._pending_feasible_refresh
        self._pending_feasible_refresh = None
        repaired = super()._rescue_committed_leg(
            observation, position, velocity, now, duration)
        if pending is None:
            return repaired
        if repaired:
            self.diagnostics['proactive_accepted'] += 1
            return True
        # This curve passed the original feasibility test in the same planner
        # call. Failed proactive optimization must not discard it or its route.
        curve, duration, now = pending
        self._select(self.route[0], curve, duration, now)
        self.diagnostics['proactive_fallbacks'] += 1
        return True
