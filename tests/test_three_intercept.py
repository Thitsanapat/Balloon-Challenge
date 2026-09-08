import unittest
import numpy as np

from BalloonPoppingGymEnv.agents.three_intercept_agent import intercept_chain
from BalloonPoppingGymEnv.agents.spline_route_agent import two_intercept_spline, boundary_curve
from BalloonPoppingGymEnv.agents.physics_guidance_agent import sample_curve
from BalloonPoppingGymEnv.agents.optimized_spiral_agent import jerk_energy


class ChainTests(unittest.TestCase):
    def test_two_segments_match_previous_solver(self):
        z = np.zeros(3)
        points = [np.array([5.,10.,30.]),np.array([30.,12.,50.])]
        previous = two_intercept_spline(z,z,z,*points,z,4.,3.)
        current = intercept_chain(z,z,z,points,z,[4.,3.])
        np.testing.assert_allclose(current,previous,atol=1e-10)

    def test_three_segments_have_continuous_position_velocity_acceleration_and_jerk(self):
        z = np.zeros(3)
        points = [np.array([10.,0.,40.]),np.array([20.,10.,50.]),np.array([30.,0.,60.])]
        times = [4.,3.,5.]
        curves = intercept_chain(z,z,z,points,z,times)
        for i in range(2):
            end = sample_curve(curves[i],times[i],times[i])
            start = sample_curve(curves[i+1],times[i+1],0.)
            np.testing.assert_allclose(end,start,atol=1e-9)
            np.testing.assert_allclose(end[0],points[i],atol=1e-9)
        np.testing.assert_allclose(sample_curve(curves[-1],times[-1],times[-1])[:3],
                                   [points[-1],z,z],atol=1e-9)

    def test_joint_fit_beats_stopping_at_every_waypoint(self):
        z = np.zeros(3)
        points = [np.array([10.,0.,40.]),np.array([20.,10.,50.]),np.array([30.,0.,60.])]
        times = [4.,3.,5.]
        curves = intercept_chain(z,z,z,points,z,times)
        stopped = [boundary_curve(p,z,z,q,z,z,t) for p,q,t in zip([z]+points[:-1],points,times)]
        self.assertLess(sum(jerk_energy(c,t) for c,t in zip(curves,times)),
                        sum(jerk_energy(c,t) for c,t in zip(stopped,times)))

    def test_invalid_durations_rejected(self):
        z = np.zeros(3)
        for times in ([0.,2.],[float('nan'),2.],[-1.,2.]):
            with self.assertRaises(ValueError):
                intercept_chain(z,z,z,[z,z],z,times)


if __name__=="__main__":
    unittest.main()
