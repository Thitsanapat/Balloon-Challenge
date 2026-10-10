import unittest
from unittest.mock import patch

from BalloonPoppingGymEnv.agents.scenario4_adaptive_entry_agent import (
    Scenario4AdaptiveEntryAgent,
)
from BalloonPoppingGymEnv.agents.scenario4_entry_horizon_agent import (
    Scenario4EntryHorizonAgent,
)
from BalloonPoppingGymEnv.agents.submission_scenario4_pure_joint_v2 import (
    Scenario4PureJointRefreshAgent,
)


class AdaptiveEntryTests(unittest.TestCase):
    def test_threshold_selects_short_entry_without_retaining_probe_state(self):
        for nominal, expected in ((8, False), (9, True)):
            with self.subTest(nominal=nominal):
                agent = object.__new__(Scenario4AdaptiveEntryAgent)
                agent.launched = False
                agent.launch_selected = False
                agent.launch_time = 42.
                agent.entry_pending = True
                agent.overlong_route_count = 9
                agent.adaptive_decided = False
                agent.launch_comparisons = []
                agent.diagnostics = {}
                actual_choices = []

                def probe(instance, observation):
                    instance.launch_comparisons = [[[0., 0., 1.], nominal, 28.]]
                    instance.launched = True
                    return {'discarded': True}

                def actual(instance, observation):
                    actual_choices.append((instance.entry_pending, instance.launched))
                    return {'launch': True}

                with patch.object(Scenario4PureJointRefreshAgent, 'get_action', probe), \
                     patch.object(Scenario4EntryHorizonAgent, 'get_action', actual):
                    self.assertEqual(agent.get_action({'simulation_time': 42.}),
                                     {'launch': True})
                    agent.get_action({'simulation_time': 42.01})
                self.assertEqual(actual_choices, [(expected, False),
                                                  (expected, False)])
                self.assertEqual(agent.diagnostics['adaptive_full_nominal_count'], nominal)
                self.assertEqual(agent.diagnostics['adaptive_short_entry'], expected)


if __name__ == '__main__':
    unittest.main()
