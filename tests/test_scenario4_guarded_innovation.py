import unittest

import numpy as np

from BalloonPoppingGymEnv.agents.scenario4_guarded_innovation_agent import (
    Scenario4GuardedInnovationAgent,
    accept_terminal_correction,
)
from BalloonPoppingGymEnv.evaluation.evaluate import load_scenario_parameters


class GuardedInnovationTests(unittest.TestCase):
    def test_never_apply_saturated_or_large_tracking_correction(self):
        self.assertTrue(accept_terminal_correction(1.0, 3.0, False))
        self.assertFalse(accept_terminal_correction(1.0, 3.0, True))
        self.assertFalse(accept_terminal_correction(3.01, 3.0, False))

    def test_gate_uses_public_balloon_radius(self):
        _, given = load_scenario_parameters(4)
        agent = Scenario4GuardedInnovationAgent(
            given, launch_candidates=0, timing_candidates=0)
        self.assertAlmostEqual(agent.innovation_max_tracking_error,
                               2 * given["balloon"]["radius"])
        self.assertEqual(agent.diagnostics["guarded_applied_steps"], 0)

    def test_invalid_gate_rejected(self):
        _, given = load_scenario_parameters(4)
        with self.assertRaises(ValueError):
            Scenario4GuardedInnovationAgent(
                given, innovation_tracking_radius_multiple=np.nan)


if __name__ == "__main__":
    unittest.main()
