import unittest

from scripts.assess_shadow_calibration import assess, fit, rows


def report(seed, predictions):
    return dict(scenario=4, seed=seed, terminated=True, truncated=False,
                scope='Evaluator-only fresh Scenario 4 shadow validation',
                anchors=[dict(target_index=7, predictions=predictions)])


def prediction(predicted=1., actual=1.8, **kwargs):
    return dict(horizon=.5, predicted_closest_distance=predicted,
                actual_closest_distance=actual, plan_changes=0,
                target_changes=0, **kwargs)


class ShadowCalibrationTests(unittest.TestCase):
    def test_fields_cannot_leak_between_training_and_testing(self):
        with self.assertRaises(ValueError):
            assess([report(1, [prediction()])], [report(1, [prediction()])])

    def test_changed_reference_or_target_is_not_fit_as_fixed_plan(self):
        changed = prediction()
        changed['plan_changes'] = 1
        self.assertEqual(rows(report(1, [changed])), [])
        self.assertEqual(len(rows(report(1, [changed]), fixed_only=False)), 1)

    def test_corrected_distance_is_fitted_only_from_training(self):
        result = assess([report(1, [prediction()]), report(2, [prediction()])],
                        [report(3, [prediction(actual=4.)])])
        self.assertAlmostEqual(result['calibration']['0.5']['bias_m'], .8)
        self.assertEqual(result['test_metrics']['0.5']['raw_false_capture_count'], 1)
        self.assertEqual(result['test_metrics']['0.5']['upper_bound_capture_predictions'], 0)
        self.assertFalse(result['controller_experiment_gate_passed'])

    def test_target_clusters_not_time_rows_weight_the_bias(self):
        data = [dict(seed=1, target=1, horizon=.5, predicted=1., actual=1.)]*100
        data += [dict(seed=2, target=2, horizon=.5, predicted=1., actual=3.)]
        self.assertAlmostEqual(fit(data)['0.5']['bias_m'], 1.)


if __name__ == '__main__':
    unittest.main()
