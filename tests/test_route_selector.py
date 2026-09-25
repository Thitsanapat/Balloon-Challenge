import unittest
from unittest.mock import patch

import numpy as np

from BalloonPoppingGymEnv.agents.route_selector_agent import (
    SELECTOR_FEATURES, SelectorChainAgent, selector_features,
)
from BalloonPoppingGymEnv.evaluation.evaluate import load_scenario_parameters


class RouteSelectorTests(unittest.TestCase):
    def setUp(self):
        _, given = load_scenario_parameters(1)
        self.agent = SelectorChainAgent(given, selector_candidates=3)
        self.agent.launch_selected = True

    def test_features_are_fixed_public_shape_and_ignore_hidden_targets(self):
        observation = {
            'balloon_states': np.array([[5., 0., 22., 0., 0., 0.],
                                        [1., 0., 22., 0., 0., 0.],
                                        [2., 0., 22., 0., 0., 0.]]),
            'balloon_status': np.array([1, 0, 1]),
            'rocket_sensors': np.r_[np.zeros(6), [0., 0., 20.], np.zeros(3)],
            'simulation_time': 25.,
        }
        saved = observation['balloon_states'].copy()
        features, mask, ids = selector_features(self.agent, observation, 3)
        self.assertEqual(features.shape, (3*SELECTOR_FEATURES+4,))
        np.testing.assert_array_equal(mask, [True, True, False])
        self.assertEqual(ids, [2, 0])
        np.testing.assert_array_equal(observation['balloon_states'], saved)

    def test_selected_rank_is_the_only_root_candidate_after_launch_choice(self):
        states = np.array([[10., 0., 20., 0., 0., 0.], [2., 0., 20., 0., 0., 0.]])
        self.agent.set_selector_action(1)
        ids = self.agent._candidate_ids(states, np.array([0, 1]), (), 0.,
                                        np.array([0., 0., 20.]), np.zeros(3), 25.)
        self.assertEqual(ids, [0])
        self.assertEqual(self.agent.diagnostics['selector_forced_first_legs'], 1)

    def test_launch_axis_probe_keeps_parent_candidates(self):
        self.agent.launch_selected = False
        states = np.zeros((2, 6))
        with patch('BalloonPoppingGymEnv.agents.submission_time_allocation_v1.ChainBeamAgent._candidate_ids',
                   return_value=[1]) as parent:
            got = self.agent._candidate_ids(states, np.array([0, 1]), (), 0.,
                                            np.zeros(3), np.zeros(3), 25.)
        self.assertEqual(got, [1])
        parent.assert_called_once()

    def test_invalid_action_rejected(self):
        for value in (-1, 3, np.nan, 1.5):
            with self.assertRaises(ValueError):
                self.agent.set_selector_action(value)

    def test_disabled_selector_retains_parent_candidates(self):
        self.agent.selector_enabled = False
        states = np.zeros((2, 6))
        with patch('BalloonPoppingGymEnv.agents.submission_time_allocation_v1.ChainBeamAgent._candidate_ids',
                   return_value=[1]) as parent:
            got = self.agent._candidate_ids(states, np.array([0, 1]), (), 0.,
                                            np.zeros(3), np.zeros(3), 25.)
        self.assertEqual(got, [1])
        parent.assert_called_once()


if __name__ == '__main__':
    unittest.main()
