import unittest
from unittest.mock import patch
import numpy as np

from BalloonPoppingGymEnv.agents.route_neighborhood_agent import RouteNeighborhoodAgent, route_neighbors
from BalloonPoppingGymEnv.agents.chain_launch_agent import ChainLaunchAgent
from BalloonPoppingGymEnv.agents.submission_sqp_v2 import batch_samples


class RouteNeighborhoodTests(unittest.TestCase):
    def setUp(self):
        self.states = np.array([[10., 0., 20., 0., 0., 0.],
                                [20., 0., 20., 0., 0., 0.],
                                [5., 0., 20., .1, 0., 0.],
                                [15., 0., 20., 0., 0., 0.]])
        self.position = np.array([0., 0., 20.])

    def test_proposals_unique_released_targets_and_bounded_times(self):
        original = self.states.copy()
        proposals = list(route_neighbors((0, 1), (4., 4.), self.states,
                                         [0, 1, 2], self.position, 9.))
        self.assertTrue(any(len(r) == 3 for r, _ in proposals))
        self.assertTrue(any(r == (1, 0) for r, _ in proposals))
        self.assertTrue(any(len(r) == 2 and 2 in r for r, _ in proposals))
        for route, times in proposals:
            self.assertEqual(len(route), len(set(route)))
            self.assertEqual(len(route), len(times))
            self.assertTrue(set(route).issubset({0, 1, 2}))
            self.assertGreater(min(times), 0.)
            self.assertLessEqual(sum(times), 9.)
        np.testing.assert_array_equal(self.states, original)

    def agent(self, budget=30):
        agent = RouteNeighborhoodAgent.__new__(RouteNeighborhoodAgent)
        agent.route_events = []
        agent.route, agent.deadlines = [0, 1], [4., 8.]
        agent.ends_v, agent.ends_a = [], []
        agent.acceleration, agent.terminal_weight = np.zeros(3), 0.
        agent.launch_time, agent.burn_time = 0., 10.
        agent.failed_until = {}
        agent.neighborhood_budget, agent.neighborhood_rounds = budget, 2
        agent.neighborhood_beam, agent.neighborhood_width = 2, 2
        agent.only_more_hits = True
        agent.diagnostics = dict.fromkeys(('neighborhood_searches', 'neighborhood_trials',
            'neighborhood_feasible', 'neighborhood_adopted', 'neighborhood_extra_planned_hits'), 0)
        agent._select = lambda *args: None
        return agent

    def test_budget_and_dense_rejection_preserve_original_route(self):
        agent = self.agent(7)
        checks = []
        def valid(curves, durations, starts, samples):
            checks.append(samples)
            return np.full(len(durations), samples != 65), *[np.zeros((len(durations), 3)) for _ in range(3)]
        agent._valid = valid
        observation = dict(balloon_states=self.states, balloon_status=np.ones(4))
        original = self.states.copy()
        with patch.object(ChainLaunchAgent, '_make_plan', side_effect=lambda *args:
                          agent.route_events.append([0., [0, 1], 8.])):
            agent._make_plan(observation, self.position, np.array([2.5, 0., 0.]), 0.)
        self.assertEqual(agent.diagnostics['neighborhood_trials'], 7)
        self.assertIn(65, checks)
        self.assertEqual(agent.route, [0, 1])
        self.assertEqual(agent.deadlines, [4., 8.])
        np.testing.assert_array_equal(self.states, original)

    def test_adopted_route_hits_predicted_centers_with_continuous_derivatives(self):
        agent = self.agent(12)
        selected = []
        def valid(curves, durations, starts, samples):
            # Accept physical constraints here to isolate optimizer bookkeeping.
            p, v, a, _ = batch_samples(curves, durations, samples)
            if samples == 65:
                selected.append((curves, durations))
            return np.ones(len(durations), dtype=bool), p[:, -1], v[:, -1], a[:, -1]
        agent._valid = valid
        with patch.object(ChainLaunchAgent, '_make_plan', side_effect=lambda *args:
                          agent.route_events.append([0., [0, 1], 8.])):
            agent._make_plan(dict(balloon_states=self.states, balloon_status=np.ones(4)),
                             self.position, np.array([2.5, 0., 0.]), 0.)
        self.assertEqual(agent.diagnostics['neighborhood_adopted'], 1)
        self.assertGreater(len(agent.route), 2)
        curves, times = selected[-1]
        p, v, a, _ = batch_samples(curves, times)
        targets = self.states[agent.route, :3]+self.states[agent.route, 3:6]*np.asarray(agent.deadlines)[:, None]
        np.testing.assert_allclose(p[:, -1], targets, atol=1e-8)
        for values in (p, v, a):
            np.testing.assert_allclose(values[:-1, -1], values[1:, 0], atol=1e-8)

    def test_disabled_search_does_not_run_optimizer(self):
        agent = self.agent(0)
        agent._solve_route = lambda *args: self.fail('disabled search must not solve')
        with patch.object(ChainLaunchAgent, '_make_plan', side_effect=lambda *args:
                          agent.route_events.append([0., [0, 1], 8.])):
            agent._make_plan(dict(balloon_states=self.states, balloon_status=np.ones(4)),
                             self.position, np.zeros(3), 0.)
        self.assertEqual(agent.route, [0, 1])
        self.assertEqual(agent.diagnostics['neighborhood_trials'], 0)


if __name__ == '__main__':
    unittest.main()
