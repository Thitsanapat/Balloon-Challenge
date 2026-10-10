import unittest

import numpy as np
from scipy.stats import ncx2

from BalloonPoppingGymEnv.agents.scenario4_chance_route_agent import (
    ForecastErrorHistory,
    ball_capture_probability,
    chance_route_score,
)
from BalloonPoppingGymEnv.agents.scenario4_near_term_chance_agent import (
    Scenario4NearTermChanceAgent,
)
from BalloonPoppingGymEnv.evaluation.evaluate import load_scenario_parameters


class NoProfile:
    def correction(self, states, status, index, horizon):
        return np.zeros(2)


class ChanceRouteTests(unittest.TestCase):
    def test_gaussian_ball_probability_matches_noncentral_chi_square(self):
        distance = np.array([0, 0.01, 0.5, 1.5, 3.0, 5.0])
        sigma = np.array([0.7, 1.0, 1.5, 1.7, 2.0, 2.5])
        actual = ball_capture_probability(distance, 1.5, sigma)
        expected = ncx2.cdf((1.5 / sigma) ** 2, 3, (distance / sigma) ** 2)
        np.testing.assert_allclose(actual, expected, rtol=1e-8, atol=1e-8)

    def test_forecast_history_uses_only_observations_after_due_time(self):
        history = ForecastErrorHistory(horizons=(1.0, 2.0), sample_period=10,
                                       peers=1, minimum_samples=1)
        state = np.array([[0., 0., 10., 1., 0., 0.]])
        status = np.array([1])
        history.observe(0., state, status, NoProfile(), 1., 10.)
        self.assertEqual(history.sample_count, 0)
        state[0, 0] = 1.5
        history.observe(1., state, status, NoProfile(), 1., 10.)
        self.assertEqual(len(history.errors[1.0]), 1)
        self.assertEqual(len(history.errors[2.0]), 0)
        np.testing.assert_allclose(history.errors[1.0][0], 0.25)
        state[0, 0] = 3.
        history.observe(2., state, status, NoProfile(), 1., 10.)
        self.assertEqual(len(history.errors[2.0]), 1)
        np.testing.assert_allclose(history.errors[2.0][0], 1.)

    def test_route_score_counts_time_aligned_incidental_once(self):
        # A straight 0->10 metre, 1-second rocket leg. Balloon 1 is near at
        # t=.5; balloon 2 visits the same position only at t=0.
        curve = np.zeros((1, 6, 3))
        curve[0, 3, 0] = 10.
        states = np.array([[10., 0., 0., 0., 0., 0.],
                           [5., 0., 0., 0., 0., 0.],
                           [5., 0., 0., 20., 0., 0.]])
        uncertainty = ForecastErrorHistory()
        both = chance_route_score(curve, [1.], states, [0, 1, 2], (0,), 1.5,
                                  uncertainty, 0.5, samples=9)
        one = chance_route_score(curve, [1.], states, [0, 1], (0,), 1.5,
                                 uncertainty, 0.5, samples=9)
        self.assertGreater(one[0], 0)
        self.assertLess(one[0], 1)
        self.assertGreater(one[1], 0)
        self.assertGreaterEqual(both[1], one[1])
        self.assertLess(both[1], one[1] + 0.03)

    def test_invalid_probability_input_rejected(self):
        with self.assertRaises(ValueError):
            ball_capture_probability(1., 1.5, 0.)

    def test_near_term_gate_leaves_long_leg_baseline_order_unchanged(self):
        _, given = load_scenario_parameters(4)
        agent = Scenario4NearTermChanceAgent(
            given, launch_candidates=0, timing_candidates=0)
        curve = np.zeros((1, 6, 3))
        curve[0, 3, 0] = 10.
        states = np.array([[10., 0., 0., 0., 0., 0.],
                           [5., 0., 0., 0., 0., 0.]])
        long = ((0,), (4.,), curve, np.zeros(3), np.zeros(3))
        short = ((0,), (2.,), curve, np.zeros(3), np.zeros(3))
        self.assertEqual(agent._node_priority(long, states, [0, 1], 42.), 4.)
        self.assertEqual(agent.diagnostics['chance_near_term_ranked_nodes'], 0)
        self.assertLess(agent._node_priority(short, states, [0, 1], 42.), 3.)
        self.assertEqual(agent.diagnostics['chance_near_term_ranked_nodes'], 1)


if __name__ == "__main__":
    unittest.main()
