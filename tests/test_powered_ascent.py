import copy
import unittest

import numpy as np

from BalloonPoppingGymEnv.agents.momentum_beam_agent import arrival_curves
from BalloonPoppingGymEnv.agents.powered_ascent_agent import PoweredAscentAgent
from BalloonPoppingGymEnv.agents.submission_sqp_v2 import batch_samples
from BalloonPoppingGymEnv.evaluation.evaluate import load_scenario_parameters


class PoweredAscentTests(unittest.TestCase):
    def test_free_arrival_hits_position_without_forcing_zero_velocity(self):
        p, v, a = np.zeros(3), np.zeros(3), np.zeros(3)
        goal = np.array([[10., 5., 20.]])
        c = arrival_curves(p, v, a, goal, np.zeros((1, 3)), np.array([3.]), [0.])
        ps, vs, acs, js = batch_samples(c, [3.])
        np.testing.assert_allclose(ps[0, 0], p)
        np.testing.assert_allclose(vs[0, 0], v)
        np.testing.assert_allclose(acs[0, 0], a)
        np.testing.assert_allclose(ps[0, -1], goal[0])
        np.testing.assert_allclose(js[0, -1], 0., atol=1e-12)
        self.assertGreater(np.linalg.norm(vs[0, -1]), 1.)

    def test_shooting_vertical_integral_matches_constant_acceleration(self):
        _, given = load_scenario_parameters(1)
        before = copy.deepcopy(given)
        agent = PoweredAscentAgent(given)
        agent.mass_flow = 0.  # Analytic constant-mass limiting case.
        agent.launch_time = 0.
        p, v = np.array([0., 0., 20.]), np.array([0., 0., 2.])
        a = np.array([0., 0., agent.thrust/agent.initial_mass-9.80665])
        duration = np.array([2.])
        target = p+v*2+a*2**2/2
        error, valid, _, _, _, vz, az = agent._shoot(
            p, v, a, target[None, :], np.zeros((1, 3)), duration, np.ones(1), 0.)
        np.testing.assert_allclose(error, 0., atol=1e-10)
        np.testing.assert_allclose(vz, v[2]+a[2]*2, atol=1e-10)
        np.testing.assert_allclose(az, a[2], atol=1e-10)
        self.assertTrue(valid[0])
        self.assertEqual(given, before)

    def test_excess_lateral_thrust_is_infeasible(self):
        _, given = load_scenario_parameters(1)
        agent = PoweredAscentAgent(given)
        _, valid, *_ = agent._shoot(np.array([0., 0., 20.]), np.zeros(3),
            np.zeros(3), np.array([[1000., 0., 30.]]), np.zeros((1, 3)),
            np.array([1.]), np.ones(1), agent.launch_time)
        self.assertFalse(valid[0])


if __name__ == '__main__':
    unittest.main()
