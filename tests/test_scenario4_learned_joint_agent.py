import unittest

from BalloonPoppingGymEnv.agents.scenario4_learned_joint_agent import (
    Scenario4LearnedJointAgent,
)
from BalloonPoppingGymEnv.agents.submission_scenario4_pure_joint_v2 import (
    Scenario4PureJointRefreshAgent,
)


class LearnedJointTests(unittest.TestCase):
    def test_route_ranking_wraps_the_selected_standalone_joint_controller(self):
        self.assertIs(Scenario4LearnedJointAgent.__base__,
                      Scenario4PureJointRefreshAgent)
        self.assertIsNot(Scenario4LearnedJointAgent._node_priority,
                         Scenario4PureJointRefreshAgent._node_priority)
        self.assertIs(Scenario4LearnedJointAgent._rescue_committed_leg,
                      Scenario4PureJointRefreshAgent._rescue_committed_leg)


if __name__ == '__main__':
    unittest.main()
