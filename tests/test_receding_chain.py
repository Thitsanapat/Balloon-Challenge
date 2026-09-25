import unittest
from unittest.mock import patch
import numpy as np
from BalloonPoppingGymEnv.agents.receding_chain_agent import RecedingChainAgent
from BalloonPoppingGymEnv.agents.chain_launch_agent import ChainLaunchAgent


class RecedingChainTests(unittest.TestCase):
    def agent(self):
        agent = RecedingChainAgent.__new__(RecedingChainAgent)
        agent.route, agent.deadlines = [0, 1], [5., 8.]
        agent.ends_v, agent.ends_a = [np.zeros(3)]*2, [np.zeros(3)]*2
        agent.route_events = [[0., [0, 1], 8.]]
        agent.plan = np.array([42.])
        agent.acceleration = np.array([0., 0., 1.])
        agent.next_route_review = 0.
        agent.route_review_interval, agent.route_review_guard = 4., 1.5
        agent.route_review_time_gain, agent.review_only_more_hits = .5, True
        agent.diagnostics = dict.fromkeys(('route_reviews', 'route_reviews_adopted',
            'route_reviews_extra_planned_hits', 'beam_searches', 'beam_candidates',
            'beam_feasible', 'maximum_depth'), 0)
        return agent

    def test_inferior_proposal_restores_controller_and_preserves_work_count(self):
        agent = self.agent()
        def parent(*args):
            if not agent.route:
                agent.route, agent.deadlines = [2], [6.]
                agent.plan[:] = -10.
                agent.acceleration[:] = -10.
                agent.route_events.append([1., [2], 5.])
                agent.diagnostics['beam_candidates'] += 123
        with patch.object(ChainLaunchAgent, '_make_plan', side_effect=parent):
            agent._make_plan(dict(balloon_status=np.ones(3)), np.zeros(3), np.zeros(3), 1.)
        self.assertEqual(agent.route, [0, 1])
        self.assertEqual(agent.deadlines, [5., 8.])
        np.testing.assert_array_equal(agent.plan, [42.])
        np.testing.assert_array_equal(agent.acceleration, [0., 0., 1.])
        self.assertEqual(len(agent.route_events), 1)
        self.assertEqual(agent.diagnostics['beam_candidates'], 123)
        self.assertEqual(agent.diagnostics['route_reviews_adopted'], 0)
        self.assertEqual(agent.next_route_review, 5.)

    def test_more_hits_adopted_but_guard_prevents_near_hit_replan(self):
        for now, adopted in ((1., 1), (4., 0)):
            agent = self.agent()
            def parent(*args):
                if not agent.route:
                    agent.route, agent.deadlines = [0, 2, 1], [5., 7., 9.]
                    agent.route_events.append([now, [0, 2, 1], 9.-now])
            with patch.object(ChainLaunchAgent, '_make_plan', side_effect=parent):
                agent._make_plan(dict(balloon_status=np.ones(3)), np.zeros(3), np.zeros(3), now)
            self.assertEqual(agent.diagnostics['route_reviews_adopted'], adopted)
            self.assertEqual(len(agent.route), 2+adopted)


if __name__ == '__main__':
    unittest.main()
