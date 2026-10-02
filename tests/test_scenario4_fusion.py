"""Focused checks for the observation-only Scenario 4 navigation estimate."""

import unittest

import numpy as np

from BalloonPoppingGymEnv.agents.scenario4_fusion_agent import (
    Scenario4FusionAgent,
    fuse_navigation_step,
)
from BalloonPoppingGymEnv.agents.submission_sqp_v2 import G
from BalloonPoppingGymEnv.evaluation.evaluate import load_scenario_parameters


class NavigationFusionTests(unittest.TestCase):
    def test_acceleration_prediction_and_gnss_gain_depend_on_accuracy(self):
        zero = np.zeros(3)
        covariance = np.zeros((3, 2, 2))
        covariance[:, 0, 0] = 1.
        covariance[:, 1, 1] = 1.
        measured_position = np.array([10., 0., 0.])
        measured_velocity = np.array([10., 0., 0.])
        kwargs = dict(position=zero, velocity=zero, covariance=covariance,
                      acceleration=np.array([2., 0., 0.]),
                      measured_position=measured_position,
                      measured_velocity=measured_velocity, dt=1.,
                      acceleration_sigma=0.8)
        uncertain = fuse_navigation_step(**kwargs, position_sigma=1e6,
                                         velocity_sigma=1e6)
        accurate = fuse_navigation_step(**kwargs, position_sigma=0.01,
                                        velocity_sigma=0.01)
        np.testing.assert_allclose(uncertain[0], [1., 0., 0.], atol=1e-8)
        np.testing.assert_allclose(uncertain[1], [2., 0., 0.], atol=1e-8)
        self.assertLess(abs(accurate[0][0] - 10.), abs(uncertain[0][0] - 10.))
        self.assertLess(abs(accurate[1][0] - 10.), abs(uncertain[1][0] - 10.))
        self.assertTrue(np.all(np.linalg.eigvalsh(accurate[2]) >= -1e-12))

    def test_body_specific_force_uses_current_attitude_and_restores_gravity(self):
        _, given = load_scenario_parameters(4)
        agent = Scenario4FusionAgent(given)
        # A 90-degree yaw maps the sensor's +x reading to world +y.
        agent.quaternion = np.array([np.sqrt(0.5), 0., 0., np.sqrt(0.5)])
        initial = np.array([0., 0., agent.elevation])
        agent.observed_acceleration = np.array([2., 0., -G[2]])
        agent.observation_time = 0.
        agent._filter_sensors(np.zeros(3), initial, np.zeros(3))
        # Make the second GNSS fix deliberately uninformative, isolating IMU
        # prediction while retaining the realistic initial covariance.
        agent.gnss_position_sigma[:] = 1e6
        agent.gnss_velocity_sigma[:] = 1e6
        agent.observation_time = 1.
        _, position, velocity = agent._filter_sensors(
            np.zeros(3), initial, np.zeros(3))
        np.testing.assert_allclose(position, initial + [0., 1., 0.], atol=1e-6)
        np.testing.assert_allclose(velocity, [0., 2., 0.], atol=1e-6)

    def test_invalid_interval_is_rejected(self):
        with self.assertRaises(ValueError):
            fuse_navigation_step(np.zeros(3), np.zeros(3), np.zeros((3, 2, 2)),
                                 np.zeros(3), np.zeros(3), np.zeros(3), 0.,
                                 np.ones(3), np.ones(3), 0.8)

    def test_agent_accepts_observation_sensor_layout(self):
        _, given = load_scenario_parameters(4)
        agent = Scenario4FusionAgent(given, launch_time=0., launch_candidates=0,
                                     timing_candidates=0, search_depth=1)
        observation = {
            'simulation_time': 0.,
            'rocket_sensors': np.r_[np.zeros(6), [0., 0., agent.elevation],
                                    np.zeros(3)],
            'balloon_states': np.zeros((100, 6)),
            'balloon_status': np.zeros(100, dtype=int),
        }
        action = agent.get_action(observation)
        self.assertTrue(action['launch'])
        self.assertTrue(np.all(np.isfinite(action['tvc'])))
        np.testing.assert_array_equal(agent.observed_acceleration, np.zeros(3))


if __name__ == '__main__':
    unittest.main()
