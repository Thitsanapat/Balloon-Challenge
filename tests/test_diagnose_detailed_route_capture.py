import unittest
from pathlib import Path

import numpy as np
import yaml

from scripts.diagnose_detailed_route_capture import (
    DETAIL_NAMES, detailed_first_leg_features,
)
from BalloonPoppingGymEnv.agents.scenario4_altitude_route_agent import (
    altitude_capture_features, FrozenAltitudeCaptureModel,
    ALTITUDE_FEATURES, Scenario4AltitudeRouteAgent,
)
from BalloonPoppingGymEnv.evaluation.evaluate import load_scenario_parameters


class DetailedCaptureFeatureTests(unittest.TestCase):
    def decision(self):
        node = np.zeros(12)
        node[:3] = (0.1, -0.2, 0.3)
        node[3:6] = (0.2, 0.1, -0.1)
        node[9:11] = (0.25, 0.5)
        context = np.zeros(9)
        context[6] = 0.8
        context[8] = 1.0
        return dict(
            target=0, time=10.0, deadline=12.0,
            features=dict(nodes=[node.tolist()], context=context.tolist(),
                          visible_mask=[True]),
            visible_states=[[15.0, -30.0, 45.0, 6.0, 3.0, -3.0]],
            acceleration=[0.0, 0.0, 3.0],
            route_metadata_aligned=True,
            ends_v=[[7.0, 4.0, -2.0]],
        )

    def test_detailed_features_include_route_and_group_context(self):
        features = detailed_first_leg_features(self.decision())
        self.assertEqual(features.shape, (len(DETAIL_NAMES),))
        self.assertTrue(np.all(np.isfinite(features)))
        self.assertAlmostEqual(features[6], 0.8 - 2.0 / 30.0)
        self.assertAlmostEqual(features[9], 0.25)
        self.assertAlmostEqual(features[10], 0.5)
        self.assertAlmostEqual(features[11], np.sqrt(3.0) / 30.0)

    def test_unaligned_route_rejected_not_silently_imputed(self):
        decision = self.decision()
        decision['route_metadata_aligned'] = False
        with self.assertRaises(ValueError):
            detailed_first_leg_features(decision)

    def test_runtime_seven_features_match_logged_decision(self):
        decision = self.decision()
        expected = detailed_first_leg_features(decision)
        observed = altitude_capture_features(
            np.asarray(decision['visible_states'][0]), np.zeros(3),
            np.zeros(3), 2.0, 20.0)
        np.testing.assert_allclose(observed, expected[[0, 1, 2, 3, 4, 5, 7]])

    def test_seed_free_frozen_model_shape(self):
        model = FrozenAltitudeCaptureModel(dict(
            schema='altitude_capture_risk_experimental_v1',
            feature_names=list(ALTITUDE_FEATURES),
            center=[0.0] * 7, scale=[1.0] * 7,
            coefficients=[0.0] * 8))
        self.assertAlmostEqual(model.probability(np.zeros(7)), 0.5)

    def test_experimental_config_constructs_without_seed(self):
        config = yaml.safe_load(Path(
            'BalloonPoppingGymEnv/evaluation/configs/'
            'scenario4_altitude_route_launch42.yaml').read_text())
        _, given = load_scenario_parameters(4)
        agent = Scenario4AltitudeRouteAgent(given, **config['agent_kwargs'])
        self.assertEqual(agent.launch_time, 42.0)
        self.assertEqual(agent.diagnostics['altitude_ranked_nodes'], 0)


if __name__ == '__main__':
    unittest.main()
