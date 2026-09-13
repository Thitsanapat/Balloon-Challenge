import unittest
from unittest.mock import patch

import numpy as np

from BalloonPoppingGymEnv.agents.capture_spline_agent import CaptureSplineAgent
from BalloonPoppingGymEnv.agents.spline_route_agent import SplineRouteAgent
from BalloonPoppingGymEnv.evaluation.evaluate import load_scenario_parameters


class CaptureSplineTests(unittest.TestCase):
    def test_shift_is_inside_balloon_and_does_not_mutate_observation(self):
        _, given = load_scenario_parameters(1)
        agent = CaptureSplineAgent(given, capture_fraction=.7)
        states = np.array([[10., 0., 20., 2., 1., 3.], [0., 0., 20., 3., 2., 1.]])
        before = states.copy()
        observation = {'balloon_states':states, 'balloon_status':np.ones(2)}
        with patch.object(SplineRouteAgent, '_make_plan', autospec=True) as planner:
            agent._make_plan(observation, np.array([0., 0., 20.]), np.zeros(3), 24.)
        shifted = planner.call_args.args[1]['balloon_states']
        np.testing.assert_array_equal(states, before)
        np.testing.assert_array_equal(shifted[:, 3:], states[:, 3:])
        distances = np.linalg.norm(shifted[:, :3]-states[:, :3], axis=1)
        self.assertTrue(np.all(distances <= agent.radius*.7+1e-9))
        self.assertAlmostEqual(shifted[0, 0], 10.-agent.radius*.7)
        self.assertTrue(np.all(np.isfinite(shifted)))

    def test_invalid_fraction_is_rejected(self):
        _, given = load_scenario_parameters(1)
        for fraction in (-.1, 1., float('nan')):
            with self.assertRaises(ValueError):
                CaptureSplineAgent(given, capture_fraction=fraction)


if __name__ == '__main__':
    unittest.main()
