import unittest
import numpy as np

from BalloonPoppingGymEnv.agents.chain_launch_agent import launch_axes


class LaunchAxesTests(unittest.TestCase):
    def test_candidates_are_unit_upward_and_observation_is_unchanged(self):
        states = np.array([[20., 0., 45., 1., 0., 2.],
                           [-30., 0., 30., 0., 0., 1.]])
        original = states.copy()
        axes = launch_axes(states, [0, 1], np.array([0., 0., 20.]), 20.)
        self.assertGreater(len(axes), 1)
        for axis in axes:
            self.assertAlmostEqual(np.linalg.norm(axis), 1.)
            self.assertGreaterEqual(axis[2], np.cos(np.radians(35.)))
        np.testing.assert_array_equal(states, original)

    def test_unreleased_targets_are_not_used(self):
        states = np.array([[20., 0., 45., 1., 0., 2.]])
        np.testing.assert_array_equal(launch_axes(states, [], np.zeros(3), 20.), [[0., 0., 1.]])
        np.testing.assert_array_equal(launch_axes(states, [0], np.zeros(3), 20., count=0), [[0., 0., 1.]])


if __name__ == '__main__':
    unittest.main()
