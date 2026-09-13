import unittest

from BalloonPoppingGymEnv.agents.scripted_spline_agent import ScriptedSplineAgent


class ScriptedTargetTests(unittest.TestCase):
    def test_skips_popped_and_waits_for_unreleased_target(self):
        agent = object.__new__(ScriptedSplineAgent)
        agent.scripted_targets = [2, 1, 0]
        agent.scripted_pointer = 0
        observation = {"balloon_status": [1, 0, 2]}
        self.assertIsNone(agent._next_scripted_target(observation))
        self.assertEqual(agent.scripted_pointer, 1)
        observation["balloon_status"][1] = 1
        self.assertEqual(agent._next_scripted_target(observation), 1)


if __name__ == "__main__":
    unittest.main()
