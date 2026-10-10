import unittest
import numpy as np
from BalloonPoppingGymEnv.agents.scenario4_intercept_candidates_agent import (
    Scenario4InterceptCandidatesAgent, intercept_proxy_scores,
)


class CandidateTests(unittest.TestCase):
    def test_time_alignment_and_momentum(self):
        states = np.array([[20., 0, 10, 0, 0, 0], [5., 0, 10, -20, 0, 0]])
        scores = intercept_proxy_scores(states, [0, 0, 10], [10, 0, 0], 0,
                                        [2., 4.], [15., 15.], [0, 0, 0], 1.2)
        self.assertLess(scores[0], scores[1])

    def test_elapsed_projects_balloon_not_rocket_twice(self):
        states = np.array([[0., 0, 10, 2, 0, 0]])
        first = intercept_proxy_scores(states, [10, 0, 10], [2, 0, 0], 5,
                                       [2.], [15.], [0, 0, 0], 1.2)
        shifted = states.copy()
        shifted[:, :3] += shifted[:, 3:] * 5
        second = intercept_proxy_scores(shifted, [10, 0, 10], [2, 0, 0], 0,
                                        [2.], [15.], [0, 0, 0], 1.2)
        np.testing.assert_array_equal(first, second)

    def make_agent(self):
        agent = Scenario4InterceptCandidatesAgent.__new__(Scenario4InterceptCandidatesAgent)
        agent.branch_targets = 2
        agent.failed_until = {3: 10}
        agent.candidate_keep_fraction = .5
        agent.candidate_time_samples = 5
        agent.launch_time = 0
        agent.burn_time = 30
        agent.leg_horizon = 10
        agent.reserve = .8
        agent.disturbance = np.zeros(3)
        agent.max_tilt = 1.2
        agent.available_acceleration = lambda now: 15.
        agent.diagnostics = dict(intercept_candidate_calls=0, intercept_candidate_changes=0)
        return agent

    def test_excludes_unreleased_visited_and_failed_without_mutation(self):
        agent = self.make_agent()
        states = np.zeros((6, 6))
        states[:, 2] = 10
        states[:, 0] = np.arange(6)
        saved = states.copy()
        ids = agent._candidate_ids(states, [0, 1, 2, 3, 4], (0,), 1,
                                   np.zeros(3), np.zeros(3), 2)
        self.assertEqual(len(ids), 2)
        self.assertTrue(set(ids).issubset({1, 2, 4}))
        np.testing.assert_array_equal(states, saved)

    def test_disabled_preserves_parent_order(self):
        agent = self.make_agent()
        agent.candidate_keep_fraction = 1
        states = np.zeros((3, 6))
        states[:, 0] = [30, 10, 20]
        self.assertEqual(agent._candidate_ids(states, [0, 1, 2], (), 0,
                                             np.zeros(3), np.zeros(3), 0), [1, 2])

    def test_rejects_bad_horizon(self):
        with self.assertRaises(ValueError):
            intercept_proxy_scores(np.zeros((1, 6)), np.zeros(3), np.zeros(3),
                                   0, [0], [12], np.zeros(3), 1.2)


if __name__ == '__main__':
    unittest.main()
