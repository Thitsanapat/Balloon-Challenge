"""Unit checks for the observation-only altitude-aware drift forecast."""

import unittest
from unittest.mock import patch

import numpy as np

from BalloonPoppingGymEnv.agents.scenario4_wind_profile_agent import (
    ObservedWindProfile, Scenario4WindProfileAgent,
)
from BalloonPoppingGymEnv.agents.time_allocation_agent import TimeAllocationAgent


def linear_profile():
    z = np.array([80., 85., 90., 95., 100., 105., 110., 115., 120.])
    states = np.zeros((len(z), 6))
    states[:, 2] = z
    states[:, 3] = 2.0 + 0.10 * (z - 100.)
    states[:, 5] = 5.0
    return states, np.ones(len(z), dtype=int)


class ObservedWindProfileTests(unittest.TestCase):
    def test_linear_profile_predicts_integrated_displacement(self):
        states, status = linear_profile()
        estimator = ObservedWindProfile()
        # At z=100 with vz=5, a 2 s rise changes drift by 1 m/s.
        np.testing.assert_allclose(estimator.correction(states, status, 4, 2.0),
                                   [1.0, 0.0], atol=1e-8)

    def test_missing_altitude_support_and_unreleased_peers_fall_back(self):
        states, status = linear_profile()
        estimator = ObservedWindProfile()
        np.testing.assert_array_equal(estimator.correction(states, status, 8, 2.0),
                                      [0.0, 0.0])
        status[:] = 0
        status[4] = 1
        np.testing.assert_array_equal(estimator.correction(states, status, 4, 2.0),
                                      [0.0, 0.0])

    def test_outlier_is_downweighted_and_forecast_is_bounded(self):
        states, status = linear_profile()
        states[0, 3] += 50.0
        estimator = ObservedWindProfile(max_position_correction=2.5)
        self.assertLess(abs(estimator.correction(states, status, 4, 2.0)[0] - 1.0), 0.25)
        states, status = linear_profile()
        estimator = ObservedWindProfile(max_position_correction=0.5)
        self.assertAlmostEqual(np.linalg.norm(
            estimator.correction(states, status, 4, 2.0)), 0.5)


class Scenario4WindProfileAgentTests(unittest.TestCase):
    def test_adjusts_drift_only_without_mutating_observation(self):
        agent = Scenario4WindProfileAgent.__new__(Scenario4WindProfileAgent)
        agent.profile_horizon = 8.0
        agent.profile_max_range = 30.0
        agent.profile_nominal_speed = 14.0
        agent.profile_blend = 1.0
        agent.wind_profile = ObservedWindProfile()
        agent.route = [4]
        agent.deadlines = [2.0]
        agent.diagnostics = {"profile_plans": 0, "profile_corrected_targets": 0}
        states, status = linear_profile()
        states[:, 0] = 8.0
        observation = {"balloon_states": states, "balloon_status": status}
        saved = states.copy()
        with patch.object(TimeAllocationAgent, "_make_plan") as parent:
            agent._make_plan(observation, [0., 0., 100.], [0., 0., 0.], 0.)
        forwarded = parent.call_args.args[0]["balloon_states"]
        np.testing.assert_array_equal(observation["balloon_states"], saved)
        np.testing.assert_array_equal(forwarded[:, :3], saved[:, :3])
        self.assertAlmostEqual(forwarded[4, 3], saved[4, 3] + 0.5)
        self.assertGreater(agent.diagnostics["profile_corrected_targets"], 0)


if __name__ == "__main__":
    unittest.main()
