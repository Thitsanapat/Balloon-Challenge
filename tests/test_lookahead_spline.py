import unittest

import numpy as np

from BalloonPoppingGymEnv.agents.lookahead_spline_agent import greedy_tail_time


class RouteLookaheadTests(unittest.TestCase):
    def test_greedy_tail_sums_nearest_neighbour_legs(self):
        points = np.array([[0, 0, 0], [3, 0, 0], [3, 4, 0], [20, 0, 0]])
        self.assertAlmostEqual(greedy_tail_time(points, 0, {0}, 2, 2.0), 3.5)

    def test_excluded_points_are_not_revisited(self):
        points = np.array([[0, 0, 0], [1, 0, 0], [4, 0, 0]])
        self.assertAlmostEqual(greedy_tail_time(points, 0, {0, 1}, 3, 1.0), 4.0)


if __name__ == "__main__":
    unittest.main()
