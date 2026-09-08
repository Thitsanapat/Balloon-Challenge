import unittest

import numpy as np

from BalloonPoppingGymEnv.agents.inclined_spline_agent import InclinedSplineAgent


class InitialAttitudeTests(unittest.TestCase):
    def test_initial_aim_selects_nearest_predicted_released_balloon(self):
        agent = object.__new__(InclinedSplineAgent)
        agent.initial_target_index = None
        agent.initial_cluster_weight = 0.0
        agent.launch_lead_time = 2.0
        agent.elevation = 10.0
        states = np.zeros((3, 6))
        states[:, :3] = [[20, 0, 10], [5, 0, 10], [1, 0, 10]]
        states[0, 3] = 1.0
        observation = {"balloon_states": states, "balloon_status": [1, 1, 0]}
        index, predicted = agent._initial_aim(observation)
        self.assertEqual(index, 1)
        np.testing.assert_allclose(predicted, [7, 0, 10])

    def test_explicit_target_is_used_only_when_released(self):
        agent = object.__new__(InclinedSplineAgent)
        agent.initial_target_index = 0
        agent.initial_cluster_weight = 0.0
        agent.launch_lead_time = 0.0
        agent.elevation = 0.0
        states = np.zeros((2, 6))
        states[:, :3] = [[100, 0, 0], [10, 0, 0]]
        index, _ = agent._initial_aim(
            {"balloon_states": states, "balloon_status": [0, 1]}
        )
        self.assertEqual(index, 1)


if __name__ == "__main__":
    unittest.main()
