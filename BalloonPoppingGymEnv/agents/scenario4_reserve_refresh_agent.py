"""Keep the configured thrust reserve when refreshing an active intercept.

The initial chain search checks its reserve through ``_valid``, while the
ordinary active-leg refresh calls ``_feasible`` without that margin. This
variant requires both checks for the active refresh only. All inputs come from
the current observation and public rocket model.
"""

import numpy as np

from BalloonPoppingGymEnv.agents.scenario4_wind_profile_agent import (
    Scenario4WindProfileAgent,
)


class Scenario4ReserveRefreshAgent(Scenario4WindProfileAgent):
    """Reject active-leg updates that consume the planner's thrust margin."""

    def __init__(self, given_parameters, **kwargs):
        super().__init__(given_parameters, **kwargs)
        self.diagnostics.update(reserve_refresh_checks=0,
                                reserve_refresh_rejected=0,
                                dense_refresh_rejected=0)

    def _accept_committed_curve(self, curve, duration, now):
        if not super()._accept_committed_curve(curve, duration, now):
            return False
        self.diagnostics['reserve_refresh_checks'] += 1
        candidate = np.asarray(curve, dtype=float)[None, :, :]
        durations = np.array([duration], dtype=float)
        starts = np.array([[now]], dtype=float)
        # At 33 samples, the only intended difference from _feasible is the
        # reserve margin. The 65-sample check matches final chain validation.
        valid, _, _, _ = self._valid(candidate, durations, starts, samples=33)
        if not bool(valid[0]):
            self.diagnostics['reserve_refresh_rejected'] += 1
            return False
        valid, _, _, _ = self._valid(candidate, durations, starts, samples=65)
        if not bool(valid[0]):
            self.diagnostics['dense_refresh_rejected'] += 1
            return False
        return True
