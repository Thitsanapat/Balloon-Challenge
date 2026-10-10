import unittest

import numpy as np

from BalloonPoppingGymEnv.agents.scenario4_sensor_coupled_agent import (
    AngularResponseEstimator, Scenario4SensorCoupledAgent,
)
from BalloonPoppingGymEnv.agents.scenario4_pure_joint_refresh_agent import (
    Scenario4PureJointRefreshAgent,
)
from BalloonPoppingGymEnv.evaluation.evaluate import load_scenario_parameters


class SensorCoupledTests(unittest.TestCase):
    def test_response_learns_excited_gain_and_respects_bounds_and_slew(self):
        estimator = AngularResponseEstimator(0.01, 0.0)
        for _ in range(3000):
            previous = estimator.gain.copy()
            estimator.update([0.5, -0.5], [0.6, -0.4])
            self.assertLessEqual(np.max(np.abs(estimator.gain - previous)), 0.000301)
        np.testing.assert_allclose(estimator.gain, [1.15, 0.85], atol=1e-3)
        self.assertLess(estimator.adapted_error_square, estimator.nominal_error_square)

    def test_no_excitation_or_invalid_measurement_does_not_learn(self):
        estimator = AngularResponseEstimator(0.01, 0.02)
        for _ in range(300):
            estimator.update([0., 0.], [1., -1.])
            estimator.update([1., 1.], [np.nan, 0.])
        np.testing.assert_array_equal(estimator.gain, [1., 1.])
        self.assertEqual(estimator.samples, 0)

    def test_matched_gyro_lag_does_not_invent_response_error(self):
        estimator = AngularResponseEstimator(0.01, 0.02)
        observed = np.zeros(2)
        for step in range(1500):
            nominal = np.array([np.sin(step * 0.02), np.cos(step * 0.02)])
            observed += estimator.gyro_gain * (nominal - observed)
            estimator.update(nominal, observed)
        np.testing.assert_allclose(estimator.gain, [1., 1.], atol=1e-12)

    def test_trapezoid_tracks_constant_acceleration_without_velocity_filter_lag(self):
        _, given = load_scenario_parameters(4)
        agent = Scenario4SensorCoupledAgent(given, navigation_trapezoid=True)
        base = Scenario4PureJointRefreshAgent(given)
        for step in range(201):
            time = step * agent.dt
            velocity = np.array([2 * time, 0., 0.])
            position = np.array([time ** 2, 0., agent.elevation])
            agent._filter_sensors(np.zeros(3), position, velocity)
            base._filter_sensors(np.zeros(3), position, velocity)
        np.testing.assert_allclose(agent.filtered_position, position, atol=1e-10)
        self.assertGreater(abs(base.filtered_position[0] - position[0]), 0.01)

    def test_disabled_modes_match_parent_and_preserve_inertia(self):
        _, given = load_scenario_parameters(4)
        child = Scenario4SensorCoupledAgent(given)
        parent = Scenario4PureJointRefreshAgent(given)
        for step in range(10):
            values = (np.full(3, step * 0.01), np.array([1., 2., 3.]),
                      np.array([0.1, 0.2, 0.3]))
            for first, second in zip(child._filter_sensors(*values),
                                     parent._filter_sensors(*values)):
                np.testing.assert_array_equal(first, second)
        args = (np.array([0., 0., 1.]), np.zeros(3), 12., np.zeros(3), 20.)
        for first, second in zip(child._attitude_action(*args), parent._attitude_action(*args)):
            np.testing.assert_array_equal(first, second)
        child.adaptive_angular_response = True
        inertia = child.inertia.copy()
        child.angular_response.gain[:] = [1.15, 0.85]
        child._attitude_action(*args)
        np.testing.assert_array_equal(child.inertia, inertia)


if __name__ == '__main__':
    unittest.main()
