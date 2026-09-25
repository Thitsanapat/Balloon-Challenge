import unittest
from pathlib import Path
import numpy as np
from tests.test_constrained_chain import model
from scripts.audit_agent_submission import source_review
from BalloonPoppingGymEnv.agents import submission_time_allocation_v1 as bundle
from BalloonPoppingGymEnv.agents.time_allocation_agent import optimize_times


class TimingBundleTests(unittest.TestCase):
    def test_source_and_parent(self):
        _, flags = source_review(Path(bundle.__file__).read_text(encoding='utf-8'))
        self.assertEqual(flags, [])
        self.assertTrue(issubclass(bundle.ChainSubmissionAgent, bundle.TimeAllocationAgent))

    def test_numerical_optimizer_matches_development(self):
        agent = model()
        agent.acceleration, agent.terminal_weight, agent.leg_horizon = np.zeros(3), 0., 10.
        states = np.array([[10., 0., 20., 1., 0., 0.], [20., 0., 20., 1., 0., 0.]])
        args = (agent, (0, 1), [4., 4.], states, np.array([0., 0., 20.]), np.array([5., 0., 0.]), 0.)
        expected, actual = optimize_times(*args), bundle.optimize_times(*args)
        self.assertIsNotNone(actual)
        for left, right in zip(expected, actual):
            np.testing.assert_allclose(left, right, rtol=1e-10, atol=1e-10)


if __name__ == '__main__':
    unittest.main()
