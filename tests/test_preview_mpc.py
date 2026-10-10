import unittest
import numpy as np

from BalloonPoppingGymEnv.agents.preview_mpc import preview_correction


class PreviewMPCTests(unittest.TestCase):
    def test_exact_nominal_reference_needs_no_correction(self):
        time = .12*np.arange(1, 9)
        p0, v0, a0 = (np.array(x, float) for x in
                       ([2., 1., 3.], [4., -2., 1.], [.5, 1., -2.]))
        p = p0[None] + time[:, None]*v0 + .5*time[:, None]**2*a0
        v = v0[None] + time[:, None]*a0
        result = preview_correction(p0, v0, a0, p, v, .12, 1.5)
        np.testing.assert_allclose(result, 0., atol=1e-10)

    def test_future_position_error_gives_bounded_directional_action(self):
        time = .12*np.arange(1, 9)
        p = np.zeros((8, 3))
        p[:, 0] = .8
        v = np.zeros((8, 3))
        result = preview_correction(np.zeros(3), np.zeros(3), np.zeros(3),
                                    p, v, .12, .5)
        self.assertGreater(result[0], 0.)
        self.assertLessEqual(np.max(np.abs(result)), .5 + 1e-9)
        np.testing.assert_allclose(result[1:], 0., atol=1e-10)

    def test_invalid_hidden_or_nonfinite_reference_rejected(self):
        p = np.zeros((8, 3))
        v = np.zeros_like(p)
        p[0, 0] = np.nan
        with self.assertRaises(ValueError):
            preview_correction(np.zeros(3), np.zeros(3), np.zeros(3), p, v, .12, 1.)
        with self.assertRaises(ValueError):
            preview_correction(np.zeros(3), np.zeros(3), np.zeros(3),
                               np.zeros_like(p), v, .12, -1.)


if __name__ == '__main__':
    unittest.main()
