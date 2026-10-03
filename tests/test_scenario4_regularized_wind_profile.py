"""Behavior checks for the isolated regularized observation-only forecast."""

import unittest
from unittest.mock import patch

import numpy as np

from BalloonPoppingGymEnv.agents.scenario4_regularized_wind_agent import (
    RegularizedObservedWindProfile,
)
from BalloonPoppingGymEnv.agents.scenario4_wind_profile_agent import (
    ObservedWindProfile,
)
from scripts.diagnose_scenario4_wind_forecast import evaluate_seed, summarize


def linear_states():
    z = np.arange(80., 125., 5.)
    states = np.zeros((len(z), 6))
    states[:, 2] = z
    states[:, 3] = 2. + .1 * (z - 100.)
    states[:, 5] = 5.
    return states, np.ones(len(z), dtype=int)


class RegularizedWindTests(unittest.TestCase):
    def test_smooth_linear_profile_retains_physical_direction(self):
        states, status = linear_states()
        offset = RegularizedObservedWindProfile().correction(
            states, status, 4, 2.,
        )
        self.assertGreater(offset[0], .5)
        self.assertLess(offset[0], 1.1)
        self.assertAlmostEqual(offset[1], 0.)

    def test_sparse_end_of_profile_keeps_supported_partial_forecast(self):
        states, status = linear_states()
        incumbent = ObservedWindProfile()
        candidate = RegularizedObservedWindProfile()
        # The target rises beyond every peer during this horizon.
        np.testing.assert_array_equal(
            incumbent.correction(states, status, 6, 3.), [0., 0.],
        )
        self.assertGreater(candidate.correction(states, status, 6, 3.)[0], .1)

    def test_outlier_and_unreleased_peers_do_not_dominate(self):
        states, status = linear_states()
        states[0, 3] += 50.
        estimate = RegularizedObservedWindProfile().correction(
            states, status, 4, 2.,
        )
        self.assertGreater(estimate[0], .3)
        self.assertLess(estimate[0], 1.5)
        status[:] = 0
        status[4] = 1
        np.testing.assert_array_equal(
            RegularizedObservedWindProfile().correction(states, status, 4, 2.),
            [0., 0.],
        )

    def test_forecast_is_bounded_and_inputs_unchanged(self):
        states, status = linear_states()
        original = states.copy()
        offset = RegularizedObservedWindProfile(
            max_position_correction=.5,
        ).correction(states, status, 4, 4.)
        self.assertLessEqual(np.linalg.norm(offset), .5 + 1e-12)
        np.testing.assert_array_equal(states, original)

    def test_diagnostic_reports_same_units_for_paired_predictions(self):
        result = summarize([1., 2.], 1)
        self.assertEqual(result["count"], 2)
        self.assertAlmostEqual(result["rmse_m"], np.sqrt(2.5))
        self.assertAlmostEqual(result["corrected_fraction"], .5)

    def test_diagnostic_scores_only_later_observations(self):
        class FakeEnvironment:
            def __init__(self, **kwargs):
                self.step_index = 0

            def observation(self):
                states, status = linear_states()
                states[:, 0] = states[:, 3] * self.step_index
                states[:, 2] += 5. * self.step_index
                return {"balloon_states": states, "balloon_status": status}

            def reset(self, seed):
                return self.observation(), {}

            def step(self, action):
                self.step_index += 1
                return self.observation(), 0, False, False, {}

            def close(self):
                pass

        parameters = {"scenario": {"random_seed": None},
                      "simulation": {"time_step": 1.}}
        with patch("scripts.diagnose_scenario4_wind_forecast.BalloonPoppingEnv",
                   FakeEnvironment), patch(
                       "scripts.diagnose_scenario4_wind_forecast.load_scenario_parameters",
                       return_value=(parameters, {})):
            result = evaluate_seed(3, start=1., until=1.,
                                   sample_interval=1., horizons=(1.,))
        group = result["by_horizon"]["1.0"]
        self.assertEqual(group["methods"]["constant_velocity"]["count"], 9)
        self.assertAlmostEqual(
            group["methods"]["constant_velocity"]["rmse_m"], 0.,
        )


if __name__ == "__main__":
    unittest.main()
