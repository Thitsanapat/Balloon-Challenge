import unittest

import numpy as np

from BalloonPoppingGymEnv.agents.spline_route_agent import boundary_curve, two_intercept_spline
from BalloonPoppingGymEnv.agents.physics_guidance_agent import sample_curve
from BalloonPoppingGymEnv.agents.optimized_spiral_agent import jerk_energy


class SplineTests(unittest.TestCase):
    def test_continuity_and_endpoints(self):
        p,v,a = np.array([0.,0.,50.]),np.array([3.,2.,1.]),np.array([.1,.2,.3])
        p1,p2,v2 = np.array([12.,5.,60.]),np.array([20.,-5.,80.]),np.array([2.,0.,4.])
        c1,c2 = two_intercept_spline(p,v,a,p1,p2,v2,4.,3.)
        start = sample_curve(c1,4.,0)
        for actual,expected in zip(start[:3],(p,v,a)):
            np.testing.assert_allclose(actual,expected,atol=1e-10)
        end1,start2 = sample_curve(c1,4.,4.),sample_curve(c2,3.,0.)
        for actual,expected in zip(end1[:3],start2[:3]):
            np.testing.assert_allclose(actual,expected,atol=1e-10)
        # Natural condition for free interior acceleration in minimum jerk.
        np.testing.assert_allclose(end1[3],start2[3],atol=1e-10)
        np.testing.assert_allclose(end1[0],p1,atol=1e-10)
        end2 = sample_curve(c2,3.,3.)
        for actual,expected in zip(end2[:3],(p2,v2,np.zeros(3))):
            np.testing.assert_allclose(actual,expected,atol=1e-10)

    def test_interior_solution_minimizes_jerk(self):
        z = np.zeros(3)
        p1,p2 = np.array([10.,3.,20.]),np.array([20.,15.,40.])
        c1,c2 = two_intercept_spline(z,z,z,p1,p2,z,3.,4.)
        _,v,a,_ = sample_curve(c1,3.,3.)
        optimum = jerk_energy(c1,3.)+jerk_energy(c2,4.)
        rng = np.random.default_rng(2)
        for _ in range(10):
            vn,an = v+rng.normal(size=3),a+rng.normal(size=3)
            first = boundary_curve(z,z,z,p1,vn,an,3.)
            second = boundary_curve(p1,vn,an,p2,z,z,4.)
            self.assertGreater(jerk_energy(first,3.)+jerk_energy(second,4.),optimum)


if __name__ == "__main__":
    unittest.main()
