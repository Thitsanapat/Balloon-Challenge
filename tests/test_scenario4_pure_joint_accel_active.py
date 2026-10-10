import unittest
from unittest.mock import patch

import numpy as np

from BalloonPoppingGymEnv.agents.scenario4_pure_joint_accel_active_agent import (
    Scenario4PureJointAccelerationAgent,
    _ActiveTargetWindowProfile,
)
from BalloonPoppingGymEnv.agents.scenario4_pure_joint_refresh_agent import (
    Scenario4PureJointRefreshAgent,
)
from BalloonPoppingGymEnv.evaluation.evaluate import load_scenario_parameters


class ActiveAccelerationForecastTests(unittest.TestCase):
    def test_correction_requires_committed_target_and_one_to_three_seconds(self):
        class Baseline:
            def correction(self, *_):
                return np.array([2.0, 0.0])

        class Observed:
            def correction(self, *_):
                return np.array([0.3, -0.15])

        profile = _ActiveTargetWindowProfile(Baseline(), Observed(), 0.75)
        for target, horizon in ((None, 2.0), (4, 2.0), (3, 0.99), (3, 3.01)):
            profile.active_target = target
            np.testing.assert_array_equal(
                profile.correction(None, None, 3, horizon), [2.0, 0.0])
        self.assertEqual(profile.additional_calls, 0)
        profile.active_target = 3
        np.testing.assert_allclose(
            0.75 * profile.correction(None, None, 3, 2.0),
            [1.8, -0.15])
        self.assertEqual(profile.additional_calls, 1)

    def test_observation_updates_before_guidance_and_initial_route_is_unmodified(self):
        _, given = load_scenario_parameters(4)
        agent = Scenario4PureJointAccelerationAgent(
            given, launch_candidates=0, timing_candidates=0, search_depth=1)
        count = int(given['balloon']['num'])
        observation = {'simulation_time': 0.0,
                       'balloon_states': np.zeros((count, 6)),
                       'balloon_status': np.zeros(count, dtype=int)}
        observation['balloon_status'][0] = 1

        # Read the estimator in the parent-call hook, not after return.
        seen = []

        def check_updated(parent_agent, received):
            seen.append(parent_agent.acceleration_forecaster.previous_time)
            return {'launch': False}

        with patch.object(Scenario4PureJointRefreshAgent, 'get_action',
                          new=check_updated):
            self.assertFalse(agent.get_action(observation)['launch'])
        self.assertEqual(seen, [0.0])

        before = []

        def choose_first(parent_agent, *_):
            before.append(parent_agent.wind_profile.active_target)
            parent_agent.route = [2]

        with patch.object(Scenario4PureJointRefreshAgent, '_make_plan',
                          new=choose_first):
            agent._make_plan(observation, np.zeros(3), np.zeros(3), 0.0)
        self.assertEqual(before, [None])
        self.assertEqual(agent.wind_profile.active_target, 2)


if __name__ == '__main__':
    unittest.main()
