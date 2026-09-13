import unittest

from BalloonPoppingGymEnv.agents.committed_spline_agent import CommittedSplineAgent


class CommittedSplineTests(unittest.TestCase):
    def test_promotes_precomputed_target_and_refreshes_continuation(self):
        agent = object.__new__(CommittedSplineAgent)
        agent.target_index = None
        agent.committed_target = 2
        agent.plan = None
        agent.route_events = []
        agent.diagnostics = {
            "committed_promotions": 0,
            "committed_fallbacks": 0,
            "commitment_updates": 0,
        }
        calls = []

        def plan_once(observation, position, velocity, now, forced_target=None):
            calls.append(forced_target)
            agent.target_index = 2
            agent.plan = object()
            agent.route_events.append([now, 2, 1, 2.0, 4.0])

        agent._plan_once = plan_once
        observation = {"balloon_status": [1, 1, 1]}
        agent._make_plan(observation, None, None, 10.0)

        self.assertEqual(calls, [2])
        self.assertEqual(agent.committed_target, 1)
        self.assertEqual(agent.diagnostics["committed_promotions"], 1)
        self.assertEqual(agent.diagnostics["commitment_updates"], 1)

    def test_discards_popped_commitment(self):
        agent = object.__new__(CommittedSplineAgent)
        agent.target_index = None
        agent.committed_target = 2
        agent.plan = object()
        agent.route_events = []
        agent.diagnostics = {
            "committed_promotions": 0,
            "committed_fallbacks": 0,
            "commitment_updates": 0,
        }
        calls = []
        agent._plan_once = lambda *args, **kwargs: calls.append(kwargs.get("forced_target"))

        agent._make_plan({"balloon_status": [1, 1, 2]}, None, None, 10.0)

        self.assertEqual(calls, [None])
        self.assertIsNone(agent.committed_target)


if __name__ == "__main__":
    unittest.main()
