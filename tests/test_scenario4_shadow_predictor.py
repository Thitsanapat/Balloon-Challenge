"""Focused tests for the observation-only first-leg shadow rollout."""

import unittest

import numpy as np

from BalloonPoppingGymEnv.agents.scenario4_shadow_predictor import (
    _advance_actuator,
    _rotate_body_to_world,
    _segment_closest,
    predict_first_leg,
)


def public_hover_parameters():
    """Small synthetic public parameter set; no simulator or seed is loaded."""
    return {
        "simulation": {"time_step": 0.01},
        "balloon": {"radius": 1.5},
        "rocket": {
            "rocket_body": {"mass": 100.0, "inertia": [10.0, 10.0, 5.0]},
            "motor": {
                "dry_mass": 0.0,
                "grain_outer_radius": 0.01,
                "grain_initial_inner_radius": 0.01,
                "grain_initial_height": 0.1,
                "grain_density": 100.0,
                "grain_number": 1,
                "burn_time": 10.0,
                "thrust_source": 980.665,
                "motor_position": -1.0,
                "nozzle_position": 0.0,
            },
            "tank": {
                "initial_liquid_mass": 0.0,
                "initial_gas_mass": 0.0,
                "liquid_mass_flow_rate_out": 0.0,
                "tank_position": 1.0,
            },
            "control": {
                "max_gimbal_angle": 15.0,
                "gimbal_rate_limit": 60.0,
                "gimbal_time_constant": 0.064,
                "throttle_range": [0.0, 1.0],
                "throttle_rate_limit": 2.0,
                "throttle_time_constant": 0.080,
                "max_roll_torque": 10.0,
                "torque_rate_limit": 20.0,
                "roll_torque_time_constant": 0.032,
            },
        },
    }


def hover_kwargs():
    curve = np.zeros((6, 3))
    curve[0, 2] = 100.0
    return dict(
        given_parameters=public_hover_parameters(),
        curve=curve,
        duration=2.0,
        now=1.0,
        launch_time=0.0,
        position=np.array([0.0, 0.0, 100.0]),
        velocity=np.zeros(3),
        quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
        angular_velocity=np.zeros(3),
        gimbal_output=np.zeros(2),
        throttle_output=1.0,
        target_position=np.array([0.0, 0.0, 100.0]),
        target_velocity=np.zeros(3),
    )


class ShadowPredictorTests(unittest.TestCase):
    def test_exact_hover_stays_on_reference(self):
        result = predict_first_leg(**hover_kwargs())
        self.assertEqual(result.steps, 200)
        self.assertLess(result.closest_distance, 1e-9)
        self.assertLess(result.max_reference_error, 1e-9)
        self.assertEqual(result.acceleration_saturated_steps, 0)
        self.assertEqual(result.burnout_steps, 0)
        self.assertTrue(result.nominal_capture)
        self.assertTrue(result.robust_capture)

    def test_infeasible_lateral_reference_is_flagged_and_missed(self):
        kwargs = hover_kwargs()
        kwargs["curve"][2, 0] = 10.0  # Exact nominal endpoint at x=10 m.
        kwargs["target_position"] = np.array([10.0, 0.0, 100.0])
        result = predict_first_leg(**kwargs)
        self.assertLess(result.nominal_closest_distance, 0.05)
        self.assertGreater(result.closest_distance, 8.0)
        self.assertGreater(result.acceleration_saturated_steps, 100)
        self.assertFalse(result.nominal_capture)

    def test_public_actuator_rate_limit_and_lag(self):
        gimbal, limited = _advance_actuator(15.0, 0.0, 0.064, 60.0,
                                            0.01, -15.0, 15.0)
        throttle, throttle_limited = _advance_actuator(1.0, 0.0, 0.080,
                                                       2.0, 0.01, 0.0, 1.0)
        self.assertAlmostEqual(gimbal, 0.6)
        self.assertAlmostEqual(throttle, 0.02)
        self.assertTrue(limited)
        self.assertTrue(throttle_limited)

    def test_swept_closest_approach_catches_crossing_between_samples(self):
        distance, fraction = _segment_closest(
            np.array([1.0, 0.0, 0.0]), np.array([-1.0, 0.0, 0.0]),
            np.zeros(3), np.zeros(3))
        self.assertAlmostEqual(distance, 0.0)
        self.assertAlmostEqual(fraction, 0.5)

    def test_uncertainty_only_widens_pointwise_distance_bounds(self):
        nominal = predict_first_leg(**hover_kwargs())
        kwargs = hover_kwargs()
        kwargs.update(initial_position_sigma=0.2, target_position_sigma=0.3,
                      target_velocity_sigma=0.1, model_acceleration_sigma=0.2)
        uncertain = predict_first_leg(**kwargs)
        self.assertAlmostEqual(nominal.closest_distance, uncertain.closest_distance)
        self.assertGreater(uncertain.three_sigma_distance_upper,
                           nominal.three_sigma_distance_upper)
        self.assertLessEqual(uncertain.three_sigma_distance_lower,
                             uncertain.closest_distance)

    def test_body_quaternion_rotation_and_invalid_inputs(self):
        yaw_90 = np.array([np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)])
        np.testing.assert_allclose(_rotate_body_to_world(
            yaw_90, np.array([1.0, 0.0, 0.0])), [0.0, 1.0, 0.0], atol=1e-12)
        kwargs = hover_kwargs()
        kwargs["curve"] = np.zeros((5, 3))
        with self.assertRaises(ValueError):
            predict_first_leg(**kwargs)
        kwargs = hover_kwargs()
        kwargs["quaternion"] = np.zeros(4)
        with self.assertRaises(ValueError):
            predict_first_leg(**kwargs)


if __name__ == "__main__":
    unittest.main()
