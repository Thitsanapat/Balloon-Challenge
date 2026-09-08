import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from BalloonPoppingGymEnv.agents.optimized_spiral_agent import OptimizedSpiralAgent, jerk_energy
from BalloonPoppingGymEnv.agents.physics_guidance_agent import quintic_intercept, sample_curve
from BalloonPoppingGymEnv.evaluation.evaluate import load_scenario_parameters


class OptimizedSpiralTests(unittest.TestCase):
    def setUp(self):
        _, given = load_scenario_parameters(1)
        self.agent = OptimizedSpiralAgent(given, initial_radius=25., start_after_first_pop=False)

    def test_jerk_integral_matches_exact_gauss_quadrature(self):
        curve = np.random.default_rng(3).normal(size=(6, 3))
        nodes, weights = np.polynomial.legendre.leggauss(3)
        for duration in [0.4, 3., 12.]:
            jerk = sample_curve(curve, duration, (nodes+1)*duration/2)[3]
            expected = float(weights@np.sum(jerk**2, axis=1)*duration/2)
            self.assertAlmostEqual(jerk_energy(curve, duration)/expected, 1.)

    def test_constraint_margins_match_existing_feasibility_checks(self):
        for distance, duration in [(0., 2.), (100., 1.), (10., 6.)]:
            curve = quintic_intercept([0, 0, 100], [0, 0, 0], [0, 0, 0],
                                      [distance, 0, 100], [0, 0, 0], duration)
            margins = self.agent._constraint_margins(curve, duration, 5.)
            self.assertEqual(bool(np.all(margins >= 0)), self.agent._feasible(curve, duration, 5.))

    def test_wide_geometry_starts_without_a_pop(self):
        obs = {"balloon_states": np.array([[10., 0, 100., 2., 0, 4.]]),
               "balloon_status": np.array([1])}
        position = np.array([0., 0, 100.])
        geometry = self.agent._spiral_geometry(obs, position, 5.)
        self.assertEqual(geometry[4], 25.)
        np.testing.assert_allclose(geometry[0]+25*geometry[2], position)
        self.assertGreater(self.agent.diagnostics["axis_tilt_degrees"], 0.)

    def test_infeasible_optimizer_result_falls_back_to_feasible_seed(self):
        obs = {"balloon_states": np.array([[5., 0, 100., 1., 0, 1.]]),
               "balloon_status": np.array([1])}
        with patch("BalloonPoppingGymEnv.agents.optimized_spiral_agent.minimize",
                   return_value=SimpleNamespace(x=np.array([0.4, 100., 100., 100.]), success=True)):
            self.agent._make_plan(obs, np.array([0., 0, 100.]), np.zeros(3), 5.)
        self.assertGreater(self.agent.diagnostics["optimizer_calls"], 0)
        self.assertEqual(self.agent.diagnostics["optimizer_improvements"], 0)
        self.assertIsNotNone(self.agent.plan)
        self.assertTrue(self.agent._feasible(self.agent.plan, self.agent.plan_duration, 5.))

    def test_real_optimizer_can_improve_a_seed_without_violating_limits(self):
        obs = {"balloon_states": np.array([[5., 0, 100., 1., 0, 1.]]),
               "balloon_status": np.array([1])}
        self.agent._make_plan(obs, np.array([0., 0, 100.]), np.zeros(3), 5.)
        self.assertGreater(self.agent.diagnostics["optimizer_improvements"], 0)
        self.assertTrue(self.agent._feasible(self.agent.plan, self.agent.plan_duration, 5.))


if __name__ == "__main__":
    unittest.main()
