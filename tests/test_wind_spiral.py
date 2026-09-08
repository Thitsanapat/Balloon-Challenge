import unittest
from unittest.mock import patch

import numpy as np

from BalloonPoppingGymEnv.agents.wind_spiral_agent import WindSpiralAgent, drift_frame, spiral_state
from BalloonPoppingGymEnv.agents.physics_guidance_agent import PhysicsGuidanceAgent, sample_curve
from BalloonPoppingGymEnv.evaluation.evaluate import load_scenario_parameters


class WindSpiralTests(unittest.TestCase):
    def test_frame_is_right_handed_and_tilt_limited(self):
        for drift in ([5, 0, 2], [0, -5, 2], [0, 0, 0]):
            axis, u, v = drift_frame(drift, np.radians(45))
            frame = np.column_stack((u, v, axis))
            np.testing.assert_allclose(frame.T@frame, np.eye(3), atol=1e-12)
            self.assertAlmostEqual(np.linalg.det(frame), 1.)
            self.assertGreaterEqual(axis[2], np.cos(np.radians(45))-1e-12)

    def test_wind_rotation_rotates_the_whole_frame(self):
        rotation = np.array([[0., -1, 0], [1, 0, 0], [0, 0, 1]])
        drift = np.array([3., 1, 4])
        for original, rotated in zip(drift_frame(drift, 0.7), drift_frame(rotation@drift, 0.7)):
            np.testing.assert_allclose(rotated, rotation@original, atol=1e-12)

    def test_vertical_control_has_zero_axis_tilt(self):
        np.testing.assert_allclose(drift_frame([5, 2, 3], 0)[0], [0, 0, 1])

    def test_spiral_derivatives_and_radius_growth(self):
        axis, u, v = drift_frame([3, 1, 4], 0.7)
        drift = np.array([3., 1, 4])
        def sample(t):
            return spiral_state(np.zeros(3), drift, u, v, 2., 0.35, 0.2, 0.4, t)
        h, t = 1e-4, 3.
        p, velocity, acceleration = sample(t)
        np.testing.assert_allclose((sample(t+h)[0]-sample(t-h)[0])/(2*h), velocity, atol=1e-8)
        np.testing.assert_allclose((sample(t+h)[1]-sample(t-h)[1])/(2*h), acceleration, atol=1e-8)
        self.assertAlmostEqual(np.linalg.norm(p-drift*t), 2.+0.35*t)
        self.assertAlmostEqual(np.dot(p-drift*t, axis), 0.)

    def test_capture_then_spiral_preserves_initial_position(self):
        _, given = load_scenario_parameters(1)
        agent = WindSpiralAgent(given)
        position, velocity = np.array([10., 20., 100.]), np.array([3., 1., 4.])
        obs = {"balloon_states": np.array([[10., 20., 100., 3., 1., 4.],
                                           [20., 30., 110., 3., 1., 4.]]),
               "balloon_status": np.array([1, 1])}
        with patch.object(PhysicsGuidanceAgent, "_make_plan") as intercept:
            agent._make_plan(obs, position, velocity, 12.)
            intercept.assert_called_once()
            self.assertIsNone(agent.spiral_start)
        obs["balloon_status"][0] = 2
        with patch.object(agent, "_feasible", return_value=True):
            agent._make_plan(obs, position, velocity, 13.)
        axis, u, v = drift_frame(velocity, agent.spiral_axis_tilt)
        np.testing.assert_allclose(agent.spiral_center+agent.initial_radius*u, position)
        p, speed, accel, _ = sample_curve(agent.plan, agent.plan_duration, 0.)
        np.testing.assert_allclose(p, position)
        np.testing.assert_allclose(speed, velocity)
        end = sample_curve(agent.plan, agent.plan_duration, agent.plan_duration)
        reserve = np.sqrt(agent.available_acceleration(13.)**2-9.80665**2)
        omega = min(agent.angular_speed, np.sqrt(0.25*reserve/(2.+0.35*21.)))
        expected = spiral_state(agent.spiral_center, velocity, u, v, 2., 0.35, 0., omega, agent.plan_duration)
        for actual, target in zip(end[:3], expected):
            np.testing.assert_allclose(actual, target, atol=1e-9)


if __name__ == "__main__":
    unittest.main()
