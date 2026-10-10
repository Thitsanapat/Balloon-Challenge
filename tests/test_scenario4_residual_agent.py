"""Scenario 4 residual model compatibility and bounded-action checks."""

import unittest
from unittest.mock import patch

import numpy as np
import yaml
from pathlib import Path

from BalloonPoppingGymEnv.agents.residual_ppo_agent import residual_features
from BalloonPoppingGymEnv.agents.scenario4_residual_agent import (
    SCENARIO4_BASE_KWARGS,
    Scenario4ResidualController,
    Scenario4ResidualPolicyAgent,
    scenario4_residual_features,
)
from BalloonPoppingGymEnv.agents.scenario4_wind_profile_agent import (
    Scenario4WindProfileAgent,
)
from BalloonPoppingGymEnv.evaluation.evaluate import load_scenario_parameters


class Scenario4ResidualTests(unittest.TestCase):
    def setUp(self):
        _, self.given = load_scenario_parameters(4)

    def test_zero_residual_matches_baseline_attitude(self):
        config = yaml.safe_load(Path(
            'BalloonPoppingGymEnv/evaluation/configs/'
            'scenario4_wind_profile_launch42.yaml').read_text(encoding='utf-8'))
        self.assertEqual(SCENARIO4_BASE_KWARGS, config['agent_kwargs'])
        baseline = Scenario4WindProfileAgent(
            self.given, **SCENARIO4_BASE_KWARGS)
        residual = Scenario4ResidualController(self.given)
        args = (np.array([0.1, 0.0, np.sqrt(0.99)]),
                np.array([0.1, 0.2, 0.0]), 10.0, np.zeros(3), 15.0)
        for _ in range(10):
            for left, right in zip(baseline._attitude_action(*args),
                                   residual._attitude_action(*args)):
                np.testing.assert_array_equal(left, right)

    def test_features_and_bounded_force(self):
        controller = Scenario4ResidualController(self.given)
        observation = dict(simulation_time=42.0,
                           rocket_sensors=np.full(12, np.nan),
                           balloon_states=np.zeros((3, 6)),
                           balloon_status=np.array([1, 0, 2]))
        features = residual_features(controller, observation)
        self.assertEqual(features.shape, (96,))
        self.assertTrue(np.all(np.isfinite(features)))
        controller.set_residual([2., 0., -2.])
        np.testing.assert_array_equal(controller.residual_action,
                                      [1., 0., -1.])
        controller.plan = np.zeros((6, 3))
        with patch.object(Scenario4WindProfileAgent, '_attitude_action',
                          return_value=None) as parent:
            controller._attitude_action(
                np.array([0., 0., 1.]), np.zeros(3), 12.,
                np.zeros(3), 12.)
        axis, _, magnitude, _, available = parent.call_args.args
        self.assertLessEqual(magnitude, available + 1e-9)
        self.assertGreaterEqual(axis[2], np.cos(controller.max_tilt) - 1e-9)

    def test_target_switch_clears_previous_residual(self):
        controller = Scenario4ResidualController(self.given)
        controller.target_index = 0
        controller.set_residual([1., 0., 0.])
        def switch_target(*_):
            controller.target_index = 1
        with patch.object(Scenario4WindProfileAgent, '_make_plan',
                          side_effect=switch_target):
            controller._make_plan({}, np.zeros(3), np.zeros(3), 50.)
        np.testing.assert_array_equal(controller.residual_action,
                                      np.zeros(3))

    def test_untrained_policy_is_zero_and_keeps_baseline_settings(self):
        controller = Scenario4ResidualPolicyAgent(self.given)
        np.testing.assert_array_equal(
            controller.policy_action(np.zeros(96, np.float32)),
            np.zeros(3))
        self.assertEqual(controller.launch_time, 42.0)
        self.assertEqual(controller.profile_blend, 0.75)

    def test_active_target_is_included_even_when_not_nearest(self):
        controller = Scenario4ResidualController(self.given)
        controller.target_index = 1
        states = np.array([
            [5., 0., 20., 0., 0., 0.],
            [30., 0., 20., 0., 0., 0.],
            [10., 0., 20., 0., 0., 0.],
        ])
        observation = dict(simulation_time=42.0,
                           rocket_sensors=np.full(12, np.nan),
                           balloon_states=states,
                           balloon_status=np.ones(3, dtype=int))
        generic = residual_features(controller, observation)
        focused = scenario4_residual_features(controller, observation)
        self.assertEqual(generic.shape, focused.shape)
        self.assertAlmostEqual(generic[32], 0.05)
        self.assertAlmostEqual(focused[32], 0.30)
        self.assertAlmostEqual(focused[40], 0.05)


if __name__ == "__main__":
    unittest.main()
