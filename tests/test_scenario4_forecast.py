"""Focused numerical checks for the observation-only gust forecast."""

import unittest
from unittest.mock import patch

import numpy as np

from BalloonPoppingGymEnv.agents.scenario4_forecast_agent import (
    BalloonAccelerationTracker,
    Scenario4ForecastAgent,
)
from BalloonPoppingGymEnv.agents.time_allocation_agent import TimeAllocationAgent


def observation(time, velocity, status=1, position=(5.0, 0.0, 20.0)):
    return {
        "simulation_time": float(time),
        "balloon_status": np.array([status]),
        "balloon_states": np.array([[*position, *velocity]], dtype=float),
    }


class BalloonAccelerationTrackerTests(unittest.TestCase):
    def test_constant_acceleration_forecasts_short_horizon(self):
        tracker = BalloonAccelerationTracker(1, smoothing_tau=0.05, min_history=0.1,
                                             max_position_correction=10.0)
        for step in range(101):
            t = step / 100.0
            tracker.update(observation(t, (2 * t, 0.0, t)))
        np.testing.assert_allclose(tracker.acceleration[0], [2.0, 0.0, 1.0], atol=1e-3)
        np.testing.assert_allclose(tracker.position_correction(0, 1.0), [1.0, 0.0, 0.5], atol=1e-3)

    def test_release_reset_and_clamps(self):
        tracker = BalloonAccelerationTracker(1, smoothing_tau=0, min_history=0,
                                             max_horizontal_acceleration=2.0,
                                             max_vertical_acceleration=3.0,
                                             max_position_correction=1.0)
        tracker.update(observation(0.0, (0.0, 0.0, 0.0)))
        tracker.update(observation(0.01, (10.0, 0.0, 10.0)))
        np.testing.assert_allclose(tracker.acceleration[0], [2.0, 0.0, 3.0])
        self.assertAlmostEqual(np.linalg.norm(tracker.position_correction(0, 2.0)), 1.0)
        tracker.update(observation(0.02, (0.0, 0.0, 0.0), status=2))
        np.testing.assert_array_equal(tracker.position_correction(0, 1.0), np.zeros(3))
        tracker.update(observation(0.03, (0.0, 0.0, 0.0)))
        np.testing.assert_array_equal(tracker.position_correction(0, 1.0), np.zeros(3))

    def test_recent_velocity_change_replaces_old_trend(self):
        tracker = BalloonAccelerationTracker(1, smoothing_tau=0.05, min_history=0.1)
        for step in range(101):
            t = step / 100.0
            tracker.update(observation(t, (2.0 * t, 0.0, 0.0)))
        self.assertGreater(tracker.acceleration[0, 0], 1.9)
        for step in range(101, 201):
            t = step / 100.0
            tracker.update(observation(t, (2.0 - 2.0 * (t - 1.0), 0.0, 0.0)))
        self.assertLess(tracker.acceleration[0, 0], -1.9)
        self.assertLess(tracker.position_correction(0, 1.0)[0], 0.0)

    def test_gap_and_duplicate_time_do_not_create_spurious_acceleration(self):
        tracker = BalloonAccelerationTracker(1, smoothing_tau=0, min_history=0)
        tracker.update(observation(0.0, (0.0, 0.0, 0.0)))
        tracker.update(observation(0.0, (100.0, 0.0, 0.0)))
        tracker.update(observation(1.0, (100.0, 0.0, 0.0)))
        np.testing.assert_array_equal(tracker.acceleration[0], np.zeros(3))
        np.testing.assert_array_equal(tracker.position_correction(0, 1.0), np.zeros(3))


class Scenario4ForecastAgentTests(unittest.TestCase):
    def test_only_near_observed_target_is_adjusted_and_original_is_unchanged(self):
        agent = Scenario4ForecastAgent.__new__(Scenario4ForecastAgent)
        agent.forecast_horizon = 1.5
        agent.forecast_max_range = 30.0
        agent.forecast_nominal_speed = 6.0
        agent.forecast_tracker = BalloonAccelerationTracker(2, smoothing_tau=0, min_history=0)
        agent.forecast_tracker.last_valid[:] = True
        agent.forecast_tracker.acceleration[:] = [2.0, 0.0, 0.0]
        agent.route = [0]
        agent.deadlines = [1.0]
        agent.diagnostics = {"forecast_plans": 0, "forecast_corrected_targets": 0}
        original = {
            "balloon_states": np.array([[5.0, 0.0, 20.0, 0.0, 0.0, 0.0],
                                        [50.0, 0.0, 20.0, 0.0, 0.0, 0.0]]),
            "balloon_status": np.array([1, 1]),
        }
        saved = original["balloon_states"].copy()
        with patch.object(TimeAllocationAgent, "_make_plan") as parent:
            agent._make_plan(original, np.array([0.0, 0.0, 20.0]), np.zeros(3), 0.0)
        forwarded = parent.call_args.args[0]["balloon_states"]
        np.testing.assert_allclose(forwarded[0, :3], [6.0, 0.0, 20.0])
        np.testing.assert_array_equal(forwarded[1], saved[1])
        np.testing.assert_array_equal(original["balloon_states"], saved)
        self.assertEqual(agent.diagnostics["forecast_corrected_targets"], 1)


if __name__ == "__main__":
    unittest.main()
