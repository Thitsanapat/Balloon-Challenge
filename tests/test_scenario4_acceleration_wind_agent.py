import unittest
from unittest.mock import patch

import numpy as np

from BalloonPoppingGymEnv.agents.scenario4_acceleration_wind_agent import (
    Scenario4AccelerationWindAgent,
    _AugmentedProfile,
)
from BalloonPoppingGymEnv.agents.scenario4_wind_profile_agent import (
    Scenario4WindProfileAgent,
)
from BalloonPoppingGymEnv.evaluation.evaluate import load_scenario_parameters


class AccelerationWindAgentTests(unittest.TestCase):
    def test_augmented_profile_preserves_parent_term_and_adds_residual_once(self):
        class Baseline:
            def correction(self, *_):
                return np.array([2., 0.])

        class Observed:
            def correction(self, *_):
                return np.array([.3, -.15])

        profile = _AugmentedProfile(Baseline(), Observed(), .75,
                                    active_only=True)
        profile.active_target = 0
        raw = profile.correction(None, None, 0, 2.)
        np.testing.assert_allclose(.75*raw, [1.8, -.15])
        self.assertEqual(profile.additional_calls, 1)
        np.testing.assert_array_equal(
            profile.correction(None, None, 1, 2.), [2., 0.])
        np.testing.assert_array_equal(
            profile.correction(None, None, 0, 4.), [2., 0.])
        self.assertEqual(profile.additional_calls, 1)

    def test_agent_updates_observed_acceleration_before_parent_action(self):
        _, given = load_scenario_parameters(4)
        agent = Scenario4AccelerationWindAgent(
            given, launch_candidates=0, timing_candidates=0, search_depth=1)
        count = given['balloon']['num']
        states = np.zeros((count, 6))
        status = np.zeros(count, dtype=int)
        status[0] = 1
        observation = {'simulation_time': 0., 'balloon_states': states,
                       'balloon_status': status}
        with patch.object(Scenario4WindProfileAgent, 'get_action',
                          return_value={'launch': False}) as parent:
            action = agent.get_action(observation)
        self.assertEqual(action['launch'], False)
        self.assertEqual(parent.call_count, 1)
        self.assertEqual(agent.acceleration_forecaster.previous_time, 0.)
        self.assertTrue(agent.acceleration_forecaster.previous_valid[0])
        self.assertFalse(agent.acceleration_forecaster.previous_valid[1])


if __name__ == '__main__':
    unittest.main()
