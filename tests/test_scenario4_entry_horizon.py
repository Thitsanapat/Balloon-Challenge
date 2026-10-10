import unittest
from unittest.mock import patch

from BalloonPoppingGymEnv.agents.scenario4_entry_horizon_agent import (
    Scenario4EntryHorizonAgent,
)
from BalloonPoppingGymEnv.agents.submission_scenario4_pure_joint_v2 import (
    Scenario4PureJointRefreshAgent,
)


class EntryHorizonTests(unittest.TestCase):
    def fake_agent(self):
        agent = object.__new__(Scenario4EntryHorizonAgent)
        agent.launched = False
        agent.launch_selected = False
        agent.launch_time = 42.
        agent.search_depth = 14
        agent.entry_search_depth = 6
        agent.entry_pending = True
        agent.route = []
        agent.diagnostics = {'entry_horizon_applied': 0, 'entry_route': []}
        return agent

    def test_entry_only_then_full_depth_restored(self):
        agent = self.fake_agent()
        depths = []

        def parent_get_action(instance, observation):
            depths.append(instance.search_depth)
            instance.launched = True
            if len(depths) == 2:
                instance.route = [7, 8]
            return {'launch': True}

        with patch.object(Scenario4PureJointRefreshAgent, 'get_action', parent_get_action):
            self.assertEqual(agent.get_action({'simulation_time': 42.}), {'launch': True})
            self.assertEqual(agent.get_action({'simulation_time': 42.01}), {'launch': True})
            self.assertEqual(agent.get_action({'simulation_time': 42.02}), {'launch': True})
        self.assertEqual(depths, [6, 6, 14])
        self.assertEqual(agent.search_depth, 14)
        self.assertFalse(agent.entry_pending)
        self.assertEqual(agent.diagnostics['entry_horizon_applied'], 1)
        self.assertEqual(agent.diagnostics['entry_route'], [7, 8])

    def test_restore_depth_on_failure(self):
        agent = self.fake_agent()
        with patch.object(Scenario4PureJointRefreshAgent, 'get_action',
                          side_effect=RuntimeError('failed')):
            with self.assertRaisesRegex(RuntimeError, 'failed'):
                agent.get_action({'simulation_time': 42.})
        self.assertEqual(agent.search_depth, 14)


if __name__ == '__main__':
    unittest.main()
