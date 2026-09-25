import unittest
from unittest.mock import patch
import numpy as np
from BalloonPoppingGymEnv.agents.group_portfolio_agent import observed_groups, portfolio_key, GroupPortfolioAgent
from BalloonPoppingGymEnv.agents.time_allocation_agent import TimeAllocationAgent


class GroupPortfolioTests(unittest.TestCase):
    def test_neighborhoods_exclude_unreleased_and_diverging_targets(self):
        states = np.zeros((5, 6))
        states[:, 0] = [0., 1., 2., 3., 1.]
        states[3, 3] = 20.
        before = states.copy()
        groups = observed_groups(states, [0, 1, 2, 3], np.zeros(3), np.zeros(3), 10., 3)
        self.assertEqual(groups, [(0, 1, 2)])
        np.testing.assert_array_equal(states, before)

    def test_singleton_route_wins_if_group_costs_hits_or_time(self):
        self.assertIsNone(portfolio_key([3, 4], [4., 5.], 3, 12., False))
        self.assertIsNone(portfolio_key([3, 4, 5], [4., 8., 13.], 3, 12., False))
        self.assertEqual(portfolio_key([3, 4, 5, 6], [4., 8., 13., 15.], 3, 12., True), (-4, 15.))
        self.assertIsNone(portfolio_key([3, 4, 5], [4., 6., 8.], 3, 12., True))
        self.assertIsNone(portfolio_key([3, 3, 3], [4., 6., 8.], 3, 12., False))

    def test_disabled_exact_parent(self):
        agent = GroupPortfolioAgent.__new__(GroupPortfolioAgent)
        agent.group_trials = 0
        with patch.object(TimeAllocationAgent, '_make_plan') as parent:
            agent._make_plan({}, None, None, 0.)
            parent.assert_called_once_with({}, None, None, 0.)

    def test_rejected_group_restores_all_baseline_control_state(self):
        agent = GroupPortfolioAgent.__new__(GroupPortfolioAgent)
        agent.group_trials, agent.group_radius, agent.group_entry_depth = 2, 12., 3
        agent.group_extra_only, agent.focus_ids = True, ()
        agent.route_events, agent.failed_until = [], {}
        agent.diagnostics = dict(group_comparisons=0, group_adopted=0, group_extra_planned_hits=0)
        def make_plan(*args):
            focused = bool(agent.focus_ids)
            agent.route = [0] if focused else [0, 1]
            agent.deadlines = [3.] if focused else [3., 5.]
            agent.ends_v, agent.ends_a = [], []
            agent.target_index = 0
            agent.plan = np.array([99. if focused else 42.])
            agent.route_events.append([0., agent.route.copy(), agent.deadlines[-1]])
            agent.diagnostics['beam_searches'] = agent.diagnostics.get('beam_searches', 0)+1
        states = np.zeros((2, 6))
        with patch.object(TimeAllocationAgent, '_make_plan', side_effect=make_plan):
            agent._make_plan(dict(balloon_states=states, balloon_status=np.ones(2)), np.zeros(3), np.zeros(3), 0.)
        self.assertEqual(agent.route, [0, 1])
        np.testing.assert_array_equal(agent.plan, [42.])
        self.assertEqual(agent.focus_ids, ())
        self.assertEqual(agent.diagnostics['group_comparisons'], 1)
        self.assertEqual(agent.diagnostics['group_adopted'], 0)
        self.assertEqual(agent.diagnostics['beam_searches'], 2)

    def test_better_group_is_adopted_and_focus_does_not_leak(self):
        agent = GroupPortfolioAgent.__new__(GroupPortfolioAgent)
        agent.group_trials, agent.group_radius, agent.group_entry_depth = 2, 12., 3
        agent.group_extra_only, agent.focus_ids = True, ()
        agent.route_events, agent.failed_until = [], {}
        agent.diagnostics = dict(group_comparisons=0, group_adopted=0, group_extra_planned_hits=0)
        def make_plan(*args):
            focused = bool(agent.focus_ids)
            agent.route = [0, 1, 2] if focused else [0, 1]
            agent.deadlines = [3., 5., 8.] if focused else [3., 5.]
            agent.ends_v, agent.ends_a = [], []
            agent.target_index = 0
            agent.plan = np.array([99. if focused else 42.])
            agent.route_events.append([0., agent.route.copy(), agent.deadlines[-1]])
            agent.diagnostics['beam_searches'] = agent.diagnostics.get('beam_searches', 0)+1
        with patch.object(TimeAllocationAgent, '_make_plan', side_effect=make_plan):
            agent._make_plan(dict(balloon_states=np.zeros((3, 6)), balloon_status=np.ones(3)), np.zeros(3), np.zeros(3), 0.)
        self.assertEqual(agent.route, [0, 1, 2])
        np.testing.assert_array_equal(agent.plan, [99.])
        self.assertEqual(agent.focus_ids, ())
        self.assertEqual(agent.diagnostics['group_adopted'], 1)
        self.assertEqual(agent.diagnostics['group_extra_planned_hits'], 1)
        self.assertEqual(agent.diagnostics['beam_searches'], 2)

    def test_focus_falls_back_after_group_or_entry_depth(self):
        agent = GroupPortfolioAgent.__new__(GroupPortfolioAgent)
        agent.focus_ids, agent.group_entry_depth = (1, 2), 2
        agent.failed_until = {2: 10.}
        args = (np.zeros((4, 6)), [0, 1, 2, 3], (), 0., np.zeros(3), np.zeros(3), 0.)
        with patch.object(TimeAllocationAgent, '_candidate_ids', side_effect=lambda s, ids, *rest: ids):
            self.assertEqual(agent._candidate_ids(*args), [1])
            modified = list(args)
            modified[2] = (1,)
            self.assertEqual(agent._candidate_ids(*modified), [0, 1, 2, 3])
            modified[2] = (0, 3)
            self.assertEqual(agent._candidate_ids(*modified), [0, 1, 2, 3])


if __name__ == '__main__':
    unittest.main()
