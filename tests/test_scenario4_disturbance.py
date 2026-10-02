"""Focused tests for observation-only IMU disturbance compensation."""

import unittest

import numpy as np

from BalloonPoppingGymEnv.agents.scenario4_disturbance_agent import (
    Scenario4DisturbanceAgent,
    filter_acceleration_residual,
)
from BalloonPoppingGymEnv.evaluation.evaluate import load_scenario_parameters


class Scenario4DisturbanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _, cls.given = load_scenario_parameters(4)

    def test_residual_subtracts_thrust_and_gravity_and_clips(self):
        estimate, raw = filter_acceleration_residual(
            np.zeros(3), [2., 0., 10.], [0., 0., 1.],
            available=10., throttle=1., dt=.01, time_constant=.2,
            limit=1.5)
        np.testing.assert_allclose(raw, [2., 0., 0.])
        np.testing.assert_allclose(estimate,
                                   [1.5 * -np.expm1(-.01 / .2), 0., 0.])

    def test_rotation_from_body_frame_uses_attitude(self):
        agent = Scenario4DisturbanceAgent(self.given, launch_candidates=0,
                                          timing_candidates=0)
        agent.quaternion = np.array([np.sqrt(.5), 0., 0., np.sqrt(.5)])
        agent.observation_time = 0.
        agent.observed_acceleration = np.zeros(3)
        initial = np.array([0., 0., agent.elevation])
        agent._filter_sensors(np.zeros(3), initial, np.zeros(3))
        agent.observation_time = agent.dt
        agent.observed_acceleration = np.array([2., 0., 0.])
        agent.previous_throttle = 0.
        agent._filter_sensors(np.zeros(3), initial, np.zeros(3))
        self.assertGreater(agent.imu_disturbance[1], 0.)
        self.assertAlmostEqual(agent.imu_disturbance[0], 0., places=10)
        self.assertEqual(agent.diagnostics['imu_disturbance_updates'], 1)

    def test_parent_action_uses_imu_estimate_and_prelaunch_ignores_it(self):
        agent = Scenario4DisturbanceAgent(
            self.given, launch_time=.02, launch_candidates=0,
            timing_candidates=0, search_depth=1)

        def observation(now, body_acceleration):
            return {
                'simulation_time': now,
                'rocket_sensors': np.r_[np.zeros(3), body_acceleration,
                                        [0., 0., agent.elevation], np.zeros(3)],
                'balloon_states': np.zeros((100, 6)),
                'balloon_status': np.zeros(100, dtype=int),
            }

        agent.get_action(observation(0., np.array([1., 0., 0.])))
        np.testing.assert_array_equal(agent.imu_disturbance, np.zeros(3))
        agent.get_action(observation(.02, np.zeros(3)))
        available = agent.available_acceleration(.02)
        action = agent.get_action(observation(
            .03, [1., 0., available * agent.previous_throttle]))
        self.assertGreater(agent.imu_disturbance[0], 0.)
        self.assertGreater(agent.disturbance[0], 0.)
        self.assertTrue(np.all(np.isfinite(action['tvc'])))


if __name__ == '__main__':
    unittest.main()
