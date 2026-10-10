import unittest

import numpy as np

from BalloonPoppingGymEnv.agents.route_attention_features import routing_features
from BalloonPoppingGymEnv.agents.scenario4_learned_route_agent import (
    FEATURE_NAMES, FrozenCaptureModel, Scenario4LearnedRouteAgent,
    capture_features,
)
from scripts.diagnose_route_capture_ml import first_leg_features


class LearnedRouteTests(unittest.TestCase):
    def test_runtime_features_match_logged_training_features(self):
        state = np.array([20., -10., 75., 2., -3., 1.])
        position = np.array([5., 1., 50.])
        velocity = np.array([3., 0., 5.])
        acceleration = np.array([0., 0., 1.])
        observation = dict(balloon_states=state.reshape(1, 6),
                           balloon_status=np.array([1]))
        encoded = routing_features(observation, position, velocity,
                                   acceleration, remaining=10.,
                                   available_acceleration=14.)
        decision = dict(target=0, time=1., deadline=3.,
                        features={key: value.tolist()
                                  for key, value in encoded.items()})
        expected = first_leg_features(decision)
        actual = capture_features(state, position, velocity, 2., 14.)
        np.testing.assert_allclose(actual, expected, atol=2e-7)

    def test_physics_priority_is_unchanged_when_penalty_is_zero(self):
        agent = object.__new__(Scenario4LearnedRouteAgent)
        agent.launched = True
        agent.risk_penalty_seconds = 0.0
        agent._risk_planning_state = None
        node = ((0, 1), (2., 3.), None, None, None)
        self.assertEqual(agent._node_priority(node, np.zeros((2, 6)),
                                              np.array([0, 1]), 42.), 5.)

    def test_frozen_model_requires_expected_feature_schema(self):
        payload = dict(schema='guarded_capture_risk_v1',
                       feature_names=list(FEATURE_NAMES),
                       center=[0.] * 6, scale=[1.] * 6,
                       coefficients=[0.] * 7)
        model = FrozenCaptureModel(payload)
        self.assertAlmostEqual(model.probability(np.zeros(6)), 0.5)
        payload['feature_names'] = list(FEATURE_NAMES[:-1])
        with self.assertRaises(ValueError):
            FrozenCaptureModel(payload)


if __name__ == '__main__':
    unittest.main()
