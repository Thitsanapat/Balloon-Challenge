import unittest

from scripts.summarize_scenario4_shadow import summarize_reports


def prediction(*, plan_changes, predicted, actual, error):
    return dict(horizon=0.5, plan_changes=plan_changes,
                target_changes=0, predicted_closest_distance=predicted,
                actual_closest_distance=actual,
                closest_distance_error=predicted - actual,
                rocket_position_error=error)


class SummarizeScenario4ShadowTests(unittest.TestCase):
    def test_separates_changed_plan_and_false_capture(self):
        report = dict(scenario=4, seed=2,
                      scope="Evaluator-only fresh Scenario 4 shadow validation.",
                      horizons=[0.5], anchors=[dict(predictions=[
                          prediction(plan_changes=0, predicted=1.2,
                                     actual=1.8, error=0.4),
                          prediction(plan_changes=1, predicted=2.0,
                                     actual=1.0, error=1.0),
                      ])])
        summary = summarize_reports([report])["horizons"]["0.5"]
        self.assertEqual(summary["all"]["count"], 2)
        self.assertEqual(summary["fixed_plan"]["count"], 1)
        self.assertEqual(summary["changed_plan_count"], 1)
        self.assertEqual(summary["fixed_plan"]["false_predicted_capture_count"], 1)
        self.assertAlmostEqual(summary["fixed_plan"]["position_error_mean_m"], 0.4)

    def test_rejects_non_diagnostic_payload(self):
        with self.assertRaisesRegex(ValueError, "shadow diagnostic"):
            summarize_reports([{"scenario": 4, "seed": 2, "horizons": [],
                                "anchors": []}])
        with self.assertRaisesRegex(ValueError, "positive"):
            summarize_reports([], capture_radius=0)


if __name__ == "__main__":
    unittest.main()
