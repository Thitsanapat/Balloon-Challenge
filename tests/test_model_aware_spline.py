import unittest
import numpy as np

from BalloonPoppingGymEnv.agents.model_aware_spline_agent import tvc_thrust_axis,quadratic_drag_gain


class ForceModelTests(unittest.TestCase):
    def test_tvc_axis_matches_active_rocketpy_signs_and_is_unit(self):
        np.testing.assert_allclose(tvc_thrust_axis([0,0]),[0,0,1])
        axis=tvc_thrust_axis([10.,-5.])
        np.testing.assert_allclose(axis[:2],[np.sin(np.radians(5)),np.sin(np.radians(10))])
        self.assertAlmostEqual(np.linalg.norm(axis),1.)

    def test_quadratic_drag_gain_recovers_coefficient(self):
        velocity=np.array([20.,-10.,5.])
        coefficient=.0004
        disturbance=-coefficient*np.linalg.norm(velocity)*velocity
        self.assertAlmostEqual(quadratic_drag_gain(disturbance,velocity),coefficient)

    def test_drag_gain_ignores_low_speed_and_clamps_non_drag_force(self):
        self.assertIsNone(quadratic_drag_gain([1,2,3],[1,0,0]))
        self.assertEqual(quadratic_drag_gain([20,0,0],[10,0,0]),0.)
        self.assertEqual(quadratic_drag_gain([-20,0,0],[10,0,0],.005),.005)


if __name__=="__main__":
    unittest.main()
