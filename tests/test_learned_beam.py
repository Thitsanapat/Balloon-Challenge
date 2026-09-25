import unittest

import numpy as np

from BalloonPoppingGymEnv.agents.learned_beam_agent import (
    LearnedBeamAgent,
    RANK_FEATURES,
)
from BalloonPoppingGymEnv.evaluation.evaluate import load_scenario_parameters


class LearnedBeamTests(unittest.TestCase):
    def setUp(self):
        _, given = load_scenario_parameters(1)
        self.given = given
        self.node = ((4,), (2.,), None, np.zeros(3), np.zeros(3))

    def test_zero_bonus_is_exact_parent_priority(self):
        agent = LearnedBeamAgent(self.given)
        agent._root_ranks = {4: 1.}
        self.assertEqual(agent._node_priority(self.node, None, None, None), 2.)

    def test_public_root_score_only_adjusts_first_target_priority(self):
        agent = LearnedBeamAgent(
            self.given, rank_coefficients=np.ones(RANK_FEATURES),
            rank_feature_mean=np.zeros(RANK_FEATURES),
            rank_feature_scale=np.ones(RANK_FEATURES), rank_bonus=.4,
        )
        agent._root_ranks = {4: .5}
        self.assertAlmostEqual(agent._node_priority(self.node, None, None, None), 1.8)
        agent._root_ranks = {}
        self.assertEqual(agent._node_priority(self.node, None, None, None), 2.)

    def test_invalid_model_vectors_are_rejected(self):
        with self.assertRaises(ValueError):
            LearnedBeamAgent(self.given, rank_coefficients=np.zeros(RANK_FEATURES-1))
        with self.assertRaises(ValueError):
            LearnedBeamAgent(self.given, rank_feature_scale=np.zeros(RANK_FEATURES))
        with self.assertRaises(ValueError):
            LearnedBeamAgent(self.given, rank_bonus=.5)

    def test_score_uses_released_observations_without_mutation(self):
        agent = LearnedBeamAgent(
            self.given, rank_coefficients=np.ones(RANK_FEATURES),
            rank_feature_mean=np.zeros(RANK_FEATURES),
            rank_feature_scale=np.ones(RANK_FEATURES), rank_bonus=.1,
        )
        observation = {
            'balloon_states': np.array([[2., 0., 30., 0., 0., 0.],
                                        [1., 0., 30., 0., 0., 0.]]),
            'balloon_status': np.array([1, 0]),
            'rocket_sensors': np.r_[np.zeros(6), [0., 0., 20.], np.zeros(3)],
            'simulation_time': 25.,
        }
        states = observation['balloon_states'].copy()
        ranks = agent._root_scores(observation)
        self.assertEqual(set(ranks), {0})
        np.testing.assert_array_equal(observation['balloon_states'], states)


if __name__ == '__main__':
    unittest.main()
