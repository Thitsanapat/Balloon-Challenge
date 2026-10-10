import copy
import unittest

from scripts.assess_guarded_route_ml_stage2 import EXPECTED_SEEDS, assess


class Stage2AssessmentTests(unittest.TestCase):
    def fixture(self, selected=4, learned=8):
        hashes = {'selected': 'a' * 64, 'learned': 'b' * 64}
        data = dict(plan=dict(seeds=list(EXPECTED_SEEDS),
                              order=['selected', 'learned'],
                              configs={name: {'agent_sha256': digest}
                                       for name, digest in hashes.items()}),
                    flights=[], complete=False)
        for seed in EXPECTED_SEEDS:
            for policy, score in (('selected', selected), ('learned', learned)):
                data['flights'].append(dict(seed=seed, policy=policy,
                                            result=dict(seed=seed, scenario=4,
                                                        agent_sha256=hashes[policy],
                                                        terminated=True, truncated=False,
                                                        agent_sources_unchanged_during_run=True,
                                                        score=score,
                                                        pop_events=[[i, float(i)]
                                                                    for i in range(score)],
                                                        diagnostics=dict(position_error_sum=10.,
                                                                         tracking_steps=20,
                                                                         saturated_steps=5))))
        return data

    def test_incomplete_report_has_no_verdict(self):
        data = self.fixture()
        data['flights'] = data['flights'][:2]
        result = assess(data)
        self.assertEqual(result['conclusion'], 'pending')
        self.assertNotIn('totals', result)
        self.assertEqual(result['paired'][0]['score_delta'], 4)

    def test_complete_pass_and_score_integrity(self):
        data = self.fixture()
        data['complete'] = True
        result = assess(data)
        self.assertEqual(result['conclusion'], 'screen_passed')
        self.assertEqual(result['aggregate_delta'], 32)
        self.assertFalse(result['automatic_promotion'])
        tampered = copy.deepcopy(data)
        tampered['flights'][0]['result']['score'] += 1
        with self.assertRaisesRegex(ValueError, 'Score and pop events'):
            assess(tampered)

    def test_regression_and_missing_complete_flights_fail(self):
        data = self.fixture(selected=4, learned=4)
        data['complete'] = True
        self.assertEqual(assess(data)['conclusion'], 'screen_failed')
        data['flights'].pop()
        with self.assertRaisesRegex(ValueError, 'claims complete'):
            assess(data)


if __name__ == '__main__':
    unittest.main()
