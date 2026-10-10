import unittest

import numpy as np
import yaml

from BalloonPoppingGymEnv.agents.scenario4_budgeted_terminal_agent import (
    Scenario4BudgetedTerminalAgent, largest_feasible_innovation,
)
from BalloonPoppingGymEnv.agents.submission_scenario4_guarded_innovation_v1 import (
    Scenario4GuardedInnovationAgent,
)
from BalloonPoppingGymEnv.evaluation.evaluate import load_scenario_parameters


class BudgetedTerminalTests(unittest.TestCase):
    def test_partial_correction_stays_inside_available_force(self):
        nominal = np.array([0., 0., 14.9])
        original = nominal.copy()
        force, fraction = largest_feasible_innovation(
            nominal, np.array([6., 0.]), 1., 1., 6., 4., 1.,
            15., np.radians(75.))
        self.assertGreater(fraction, 0.05)
        self.assertLess(fraction, 1.)
        self.assertGreater(force[0], 0.)
        self.assertLessEqual(np.linalg.norm(force), 15. + 1e-8)
        np.testing.assert_array_equal(nominal, original)

    def test_full_correction_matches_unsaturated_case(self):
        force, fraction = largest_feasible_innovation(
            np.array([0., 0., 10.]), np.array([1., 0.]), 2., 0.5,
            6., 4., 1., 15., np.radians(75.))
        self.assertEqual(fraction, 1.)
        self.assertGreater(force[0], 0.)

    def test_candidate_uses_selected_public_configuration(self):
        _, given = load_scenario_parameters(4)
        with open('BalloonPoppingGymEnv/evaluation/configs/scenario4_budgeted_terminal_launch42.yaml',
                  encoding='utf-8') as handle:
            config = yaml.safe_load(handle)
        agent = Scenario4BudgetedTerminalAgent(given, **config['agent_kwargs'])
        self.assertEqual(agent.launch_time, 42.)
        self.assertEqual(agent.diagnostics['budgeted_partial_steps'], 0)

    def test_no_innovation_leaves_low_level_commands_identical(self):
        _, given = load_scenario_parameters(4)
        baseline = Scenario4GuardedInnovationAgent(given)
        candidate = Scenario4BudgetedTerminalAgent(given)
        axis = np.array([0., 0., 1.])
        jerk = np.zeros(3)
        gyro = np.zeros(3)
        original = baseline._attitude_action(axis, jerk, 10., gyro, 15.)
        changed = candidate._attitude_action(axis, jerk, 10., gyro, 15.)
        for left, right in zip(original, changed):
            np.testing.assert_allclose(left, right)


if __name__ == '__main__':
    unittest.main()
