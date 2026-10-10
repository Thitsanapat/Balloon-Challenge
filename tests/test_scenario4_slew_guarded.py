import unittest

import numpy as np

from BalloonPoppingGymEnv.agents.scenario4_slew_guarded_agent import (
    Scenario4SlewGuardedAgent,
    slew_limited_correction,
)
from BalloonPoppingGymEnv.evaluation.evaluate import load_scenario_parameters


class SlewGuardedTests(unittest.TestCase):
    def test_caps_vector_change_without_rotating_request(self):
        correction, limited = slew_limited_correction(
            np.array([0.0, 0.0]), np.array([3.0, 4.0]), 10.0, 0.1)
        np.testing.assert_allclose(correction, [0.6, 0.8])
        self.assertTrue(limited)
        correction, limited = slew_limited_correction(
            correction, np.array([0.6, 0.8]), 10.0, 0.1)
        np.testing.assert_allclose(correction, [0.6, 0.8])
        self.assertFalse(limited)

    def test_invalid_inputs_are_rejected(self):
        with self.assertRaises(ValueError):
            slew_limited_correction([0, 0], [1, 0], 0, 0.01)
        with self.assertRaises(ValueError):
            slew_limited_correction([0, 0], [np.nan, 0], 1, 0.01)

    def test_agent_uses_only_public_parameters(self):
        _, given = load_scenario_parameters(4)
        agent = Scenario4SlewGuardedAgent(
            given, launch_candidates=0, timing_candidates=0)
        self.assertEqual(agent.innovation_slew_rate, 8.0)
        self.assertEqual(agent.diagnostics["slew_limited_steps"], 0)


if __name__ == "__main__":
    unittest.main()
