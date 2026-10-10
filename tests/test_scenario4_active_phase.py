import unittest

import numpy as np

from BalloonPoppingGymEnv.agents.scenario4_active_phase_agent import (
    Scenario4ActivePhaseAgent,
    choose_active_capture_phase,
)
from BalloonPoppingGymEnv.evaluation.evaluate import load_scenario_parameters


class ZeroWind:
    def correction(self, states, status, index, horizon):
        return np.zeros(2)


class ActivePhaseTests(unittest.TestCase):
    def make_case(self, target):
        # x(t)=z(t)=t over a two-second normalized polynomial spline.
        curve = np.array([[0., 0., 0.], [2., 0., 2.]])
        states = np.array([target], dtype=float)
        return (curve, 2., 0., states[0], states,
                np.array([1]), 0, ZeroWind(), 0.75,
                1.5, 0.75, 0.25, 4., 3., 0.15)

    def test_selects_earlier_vertical_crossing(self):
        t, advanced = choose_active_capture_phase(
            *self.make_case([1., 0., 1., 0., 0., 0.]))
        self.assertTrue(advanced)
        self.assertAlmostEqual(t, 1.5)

    def test_retains_endpoint_when_target_is_there(self):
        t, advanced = choose_active_capture_phase(
            *self.make_case([2., 0., 2., 0., 0., 0.]))
        self.assertFalse(advanced)
        self.assertAlmostEqual(t, 2.0)

    def test_long_horizon_is_unchanged(self):
        args = list(self.make_case([1., 0., 1., 0., 0., 0.]))
        args[-2] = 1.0
        t, advanced = choose_active_capture_phase(*args)
        self.assertEqual(t, 2.0)
        self.assertFalse(advanced)

    def test_uses_public_radius_and_valid_settings(self):
        _, given = load_scenario_parameters(4)
        agent = Scenario4ActivePhaseAgent(
            given, launch_candidates=0, timing_candidates=0)
        self.assertEqual(agent.phase_max_horizon, 3.0)
        with self.assertRaises(ValueError):
            Scenario4ActivePhaseAgent(given, phase_step=0)


if __name__ == "__main__":
    unittest.main()
