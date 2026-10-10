import unittest

import numpy as np

from scripts.diagnose_route_capture_ml import (
    first_leg_features, fit_regularized_logistic, held_out_report,
    predict_logistic,
)


class CaptureCalibrationTests(unittest.TestCase):
    def test_executed_visible_target_can_be_temporarily_blocked(self):
        decision = dict(target=0, deadline=2.0, time=1.0,
                        features=dict(nodes=[[0.1] * 12], context=[0.1] * 9,
                                      visible_mask=[True], candidate_mask=[False]))
        features = first_leg_features(decision)
        self.assertEqual(features.shape, (6,))
        self.assertTrue(np.all(np.isfinite(features)))

    def test_constant_prior_and_finite_fit(self):
        x = np.zeros((6, 6))
        y = np.array([0, 0, 0, 1, 1, 1])
        model = fit_regularized_logistic(x, y)
        predicted = predict_logistic(x, model)
        np.testing.assert_allclose(predicted, np.full(6, 0.5), atol=1e-8)

    def test_held_out_group_never_trains_on_its_own_labels(self):
        records = []
        for seed, labels in ((10, [0, 0]), (11, [1, 1]),
                             (12, [0, 1])):
            for attempt_id, hit in enumerate(labels):
                records.append(dict(seed=seed, attempt_id=attempt_id,
                                    hit=hit, features=np.full(6, seed / 10)))
        result = held_out_report(records)
        self.assertEqual(result['fields'], 3)
        fold_10 = next(row for row in result['folds'] if row['seed'] == 10)
        self.assertEqual(fold_10['observed'], [0, 0])
        self.assertAlmostEqual(fold_10['prior'][0], 0.7)


if __name__ == '__main__':
    unittest.main()
