"""Pure and fake-environment tests; never start an official episode."""

import unittest

import numpy as np

from BalloonPoppingGymEnv.agents.scenario4_shadow_predictor import _sample_curve
from scripts.validate_scenario4_shadow import run_episode, window_curve
from tests.test_scenario4_shadow_predictor import public_hover_parameters


class FakeEnvironment:
    """Simple straight flight with a deliberately separate offline truth dict."""

    def __init__(self):
        self.step_number = 0
        self.rocket_launched = False

    def _observation(self):
        time = 0.25 * self.step_number
        sensors = np.r_[np.zeros(6), [time, 0.0, 100.0], [1.0, 0.0, 0.0]]
        return {
            "simulation_time": time,
            "rocket_sensors": sensors,
            "balloon_status": np.array([1]),
            "balloon_states": np.array([[10.0, 0.0, 100.0,
                                          0.0, 0.0, 0.0]]),
        }

    def _info(self):
        time = 0.25 * self.step_number
        return {"rocket_states": np.r_[time, 0.0, 100.0, np.zeros(10)],
                "popped_count": 0}

    def reset(self, seed):
        self.step_number = 0
        self.rocket_launched = False
        return self._observation(), self._info()

    def step(self, action):
        self.step_number += 1
        self.rocket_launched = True
        return (self._observation(), 0, self.step_number >= 16, False,
                self._info())


class FakeAgent:
    def __init__(self):
        self.received_info = False
        self.previous_tvc = np.zeros(2)
        self.previous_throttle = 1.0
        self.previous_roll = 0.0
        self.filtered_position = None
        self.filtered_velocity = None
        self.filtered_gyro = None
        self.quaternion = np.array([1.0, 0.0, 0.0, 0.0])
        self.disturbance = np.zeros(3)
        self.launch_time = 0.25
        self.tracking_frequency = 1.3
        self.attitude_frequency = 4.0
        self.max_tilt = np.radians(75.0)
        self.max_axis_rate = 1.2
        self.target_index = 0
        self.plan = None
        self.plan_start = 0.0
        self.plan_duration = 4.0

    def get_action(self, observation):
        # Any leakage of official info into the agent would fail this test.
        if "rocket_states" in observation or "popped_count" in observation:
            self.received_info = True
            raise AssertionError("Offline truth leaked to agent")
        sensors = observation["rocket_sensors"]
        self.filtered_position = sensors[6:9].copy()
        self.filtered_velocity = sensors[9:12].copy()
        self.filtered_gyro = np.zeros(3)
        self.plan_start = float(observation["simulation_time"])
        self.plan = np.zeros((6, 3))
        self.plan[0] = self.filtered_position
        self.plan[1] = self.filtered_velocity * self.plan_duration
        return {"launch": True}


class ShadowValidationTests(unittest.TestCase):
    def test_window_curve_preserves_reference_and_derivatives(self):
        rng = np.random.default_rng(12)
        curve = rng.normal(size=(6, 3))
        window = window_curve(curve, 7.0, 2.0, 3.0)
        for local_time in (0.0, 0.5, 1.5, 3.0):
            before = _sample_curve(curve, 7.0, 2.0 + local_time)
            after = _sample_curve(window, 3.0, local_time)
            for original, shifted in zip(before, after):
                np.testing.assert_allclose(original, shifted, atol=1e-10)
        with self.assertRaises(ValueError):
            window_curve(curve, 7.0, 5.0, 3.0)

    def test_runner_compares_future_truth_without_leaking_it_to_agent(self):
        env = FakeEnvironment()
        agent = FakeAgent()
        uncertainty = dict(initial_position_sigma=0.0,
                           target_position_sigma=0.0,
                           target_velocity_sigma=0.0,
                           model_acceleration_sigma=0.0)
        report = run_episode(env, agent, public_hover_parameters(), 123,
                             [0.5], [0.25, 1.0, 2.0], uncertainty)
        self.assertFalse(agent.received_info)
        self.assertEqual(report["scenario"], 4)
        self.assertEqual(report["launch_time"], 0.25)
        self.assertEqual(len(report["anchors"]), 1)
        predictions = report["anchors"][0]["predictions"]
        self.assertEqual(len(predictions), 3)
        for result in predictions:
            self.assertLess(result["rocket_position_error"], 1e-9)
            self.assertLess(abs(result["closest_distance_error"]), 1e-9)
            self.assertEqual(result["target_changes"], 0)
        self.assertGreater(predictions[2]["plan_changes"], 0)


if __name__ == "__main__":
    unittest.main()
