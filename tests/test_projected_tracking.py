import unittest
import numpy as np
from BalloonPoppingGymEnv.agents.projected_tracking_agent import project_thrust
from BalloonPoppingGymEnv.agents.submission_sqp_v2 import allocate_acceleration


class ProjectedThrustTests(unittest.TestCase):
    def test_limits_identity_and_no_mutation(self):
        tilt = np.radians(60.)
        for request in ([1., 2., 10.], [30., 20., 20.], [10., 0., -20.], [5., 2., -1.], [0., 0., -1.]):
            original = np.asarray(request)
            saved = original.copy()
            result = project_thrust(original, 15., tilt)
            norm = np.linalg.norm(result)
            self.assertLessEqual(norm, 15.+1e-10)
            self.assertGreaterEqual(result[2], norm*np.cos(tilt)-1e-10)
            np.testing.assert_array_equal(original, saved)
        np.testing.assert_array_equal(project_thrust([1., 2., 10.], 15., tilt), [1., 2., 10.])
        np.testing.assert_array_equal(project_thrust([30., 20., 20.], 0., tilt), np.zeros(3))

    def test_projection_error_is_no_worse_than_vertical_priority(self):
        # Deterministic synthetic force grid, unrelated to any simulator seed.
        tilt = np.radians(75.)
        for x in (0., 5., 15., 30.):
            for y in (-10., 0., 20.):
                for z in (-20., -1., 5., 15., 25.):
                    requested = np.array([x, y, z])
                    projected = project_thrust(requested, 14., tilt)
                    previous = allocate_acceleration(requested, 14., tilt)
                    self.assertLessEqual(np.linalg.norm(projected-requested),
                                         np.linalg.norm(previous-requested)+1e-10)


if __name__ == '__main__':
    unittest.main()
