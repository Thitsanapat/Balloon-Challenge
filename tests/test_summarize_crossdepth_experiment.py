import unittest

from scripts.summarize_crossdepth_experiment import summarize


def flight(seed, score):
    return dict(seed=seed, scenario=4, terminated=True, truncated=False,
                agent_file_unchanged_during_run=True,
                agent_sources_unchanged_during_run=True,
                config=dict(agent_class_name='Scenario4CrossDepthValueAgent'),
                score=score, pop_events=list(range(score)), diagnostics={},
                wall_seconds=200., post_reset_wall_seconds=80.)


class CrossDepthSummaryTests(unittest.TestCase):
    def test_tie_does_not_pass_development_screen(self):
        self.assertFalse(summarize([flight(0, 6), flight(2, 4)])['fresh_holdout_eligible'])

    def test_one_score_gain_cannot_hide_large_regression(self):
        self.assertFalse(summarize([flight(0, 3), flight(2, 8)])['fresh_holdout_eligible'])

    def test_incomplete_or_repeated_field_fails_closed(self):
        with self.assertRaises(ValueError):
            summarize([flight(0, 6), flight(0, 6)])
        incomplete = flight(2, 4)
        incomplete['truncated'] = True
        with self.assertRaises(ValueError):
            summarize([flight(0, 6), incomplete])


if __name__ == '__main__':
    unittest.main()
