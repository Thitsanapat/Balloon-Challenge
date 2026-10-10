import unittest

import numpy as np

from BalloonPoppingGymEnv.agents.scenario4_acceleration_forecast import (
    ObservedAccelerationForecast,
    decaying_acceleration_displacement,
)


class AccelerationForecastTests(unittest.TestCase):
    def test_decayed_acceleration_has_correct_short_and_long_limits(self):
        a = np.array([2., 0.])
        np.testing.assert_array_equal(
            decaying_acceleration_displacement(a, 0., 1.), [0., 0.])
        short = decaying_acceleration_displacement(a, .01, 1.)
        self.assertAlmostEqual(short[0], .0001, places=6)
        long = decaying_acceleration_displacement(a, 3., 1.)
        self.assertGreater(long[0], 0.)
        self.assertLess(long[0], 2. * 3.)

    def test_uses_only_prior_observed_velocity(self):
        class FlatProfile:
            def correction(self, *_):
                return np.zeros(2)

        model = ObservedAccelerationForecast(
            1, filter_seconds=.01, strength=1., decay_seconds=1.)
        states = np.array([[0., 0., 100., 0., 0., 1.]])
        model.observe(0., states, [1])
        states[0, 3] = 1.
        model.observe(1., states, [1])
        self.assertAlmostEqual(model.acceleration[0, 0], 1.)
        correction = model.correction(states, [1], 0, 2., FlatProfile())
        self.assertGreater(correction[0], 0.)
        model.observe(2., states, [0])
        np.testing.assert_array_equal(model.acceleration, np.zeros((1, 2)))
        np.testing.assert_array_equal(
            model.correction(states, [0], 0, 2., FlatProfile()), np.zeros(2))

    def test_subtracts_acceleration_already_covered_by_wind_profile(self):
        class ShearProfile:
            def correction(self, _, __, ___, horizon):
                return np.array([.5*horizon*horizon, 0.])

        model = ObservedAccelerationForecast(1, filter_seconds=.01)
        states = np.array([[0., 0., 100., 0., 0., 1.]])
        model.observe(0., states, [1])
        states[0, 3] = 1.
        model.observe(1., states, [1])
        np.testing.assert_allclose(
            model.correction(states, [1], 0, 2., ShearProfile()),
            np.zeros(2), atol=1e-12)


if __name__ == '__main__':
    unittest.main()
