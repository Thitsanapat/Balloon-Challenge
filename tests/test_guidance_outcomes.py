import unittest
from scripts.summarize_guidance_outcomes import summarize


class OutcomeTests(unittest.TestCase):
    def test_overlapping_proposals_do_not_double_count_or_become_failures(self):
        row = dict(seed=0, score=2, truncated=False,
            route_events=[[0, [1, 2, 3], 5], [2, [2, 4], 3]],
            target_events=[[0, 1], [1, 2], [2, 1]], pop_events=[[1, 1], [2, 3]],
            diagnostics={'tracking_steps': 10, 'position_error_sum': 2.})
        result = summarize(row)
        self.assertEqual(result['unique_planned'], 4)
        self.assertEqual(result['unique_selected'], 2)
        self.assertEqual(result['planned_but_never_selected'], [3, 4])
        self.assertEqual(result['selected_not_popped'], [])
        self.assertAlmostEqual(result['mean_sampled_tracking_error'], .2)
        self.assertFalse(result['telemetry_available'])


if __name__ == '__main__':
    unittest.main()
