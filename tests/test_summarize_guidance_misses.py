import unittest

from scripts.summarize_guidance_misses import summarize_flight


class GuidanceMissSummaryTests(unittest.TestCase):
    def test_vector_projection_and_best_episode_per_target(self):
        closest = dict(distance=2., time_to_deadline=0.,
                       tracking_vector=[1., 0., 0.],
                       alignment_vector=[2., 0., 0.],
                       prediction_error_vector=[-1., 0., 0.],
                       command={'allocator_saturated': True})
        flight = dict(seed=5, score=0, guidance_telemetry=dict(target_episodes=[
            dict(target=3, end_reason='target_changed', true_closest=closest),
            dict(target=3, end_reason='target_changed', true_closest=dict(
                closest, distance=3.)),
            dict(target=4, end_reason='popped', true_closest=closest),
        ]))
        result = summarize_flight(flight)
        self.assertEqual(len(result['near_misses']), 1)
        miss = result['near_misses'][0]
        self.assertAlmostEqual(miss['margin_outside_balloon_m'], .5)
        self.assertAlmostEqual(miss['tracking_along_miss_m'], 1.)
        self.assertAlmostEqual(miss['alignment_along_miss_m'], 2.)
        self.assertAlmostEqual(miss['prediction_error_along_miss_m'], -1.)
        self.assertTrue(miss['allocator_saturated_at_sample'])


if __name__ == '__main__':
    unittest.main()
