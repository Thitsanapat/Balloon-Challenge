"""Composition checks for the Scenario 4 wind and navigation experiment."""

import unittest

import numpy as np

from BalloonPoppingGymEnv.agents.scenario4_fusion_agent import Scenario4FusionAgent
from BalloonPoppingGymEnv.agents.scenario4_wind_fusion_agent import Scenario4WindFusionAgent
from BalloonPoppingGymEnv.agents.scenario4_wind_profile_agent import Scenario4WindProfileAgent
from BalloonPoppingGymEnv.evaluation.evaluate import load_scenario_parameters


class Scenario4WindFusionTests(unittest.TestCase):
    def test_composition_initializes_both_estimators(self):
        _, given = load_scenario_parameters(4)
        agent = Scenario4WindFusionAgent(given, launch_candidates=0,
                                         timing_candidates=0)
        self.assertIsInstance(agent, Scenario4FusionAgent)
        self.assertIsInstance(agent, Scenario4WindProfileAgent)
        self.assertEqual(agent.diagnostics['profile_corrected_targets'], 0)
        self.assertEqual(agent.navigation_covariance.shape, (3, 2, 2))

    def test_first_action_uses_only_observation_shape(self):
        _, given = load_scenario_parameters(4)
        agent = Scenario4WindFusionAgent(given, launch_time=0.,
                                         launch_candidates=0,
                                         timing_candidates=0,
                                         search_depth=1)
        observation = {
            'simulation_time': 0.,
            'rocket_sensors': np.r_[np.zeros(6), [0., 0., agent.elevation],
                                    np.zeros(3)],
            'balloon_states': np.zeros((100, 6)),
            'balloon_status': np.zeros(100, dtype=int),
        }
        action = agent.get_action(observation)
        self.assertTrue(action['launch'])
        self.assertTrue(np.all(np.isfinite(action['tvc'])))


if __name__ == '__main__':
    unittest.main()
