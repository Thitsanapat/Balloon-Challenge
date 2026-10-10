import unittest

from scripts.audit_route_continuation import (
    actuator_features, attempt_examples, summarize,
)


def decision(attempt, time, target, route):
    return dict(kind="decision", attempt_id=attempt, time=time, target=target,
                route_metadata_aligned=True,
                route_prefixes=[dict(targets=route[:length],
                                     durations=[1.0] * length)
                                for length in range(1, len(route) + 1)],
                ends_v=[[1., 0., 0.], [0., 1., 0.]])


def outcome(attempt, time, result):
    return dict(kind="outcome", attempt_id=attempt, time=time, outcome=result)


class RouteContinuationTests(unittest.TestCase):
    def test_actuator_geometry_uses_observed_axis_and_gyro(self):
        features = actuator_features(dict(estimated_quaternion=[1., 0., 0., 0.],
            desired_axis=[0., 1., 0.], gyro=[0., 3., 4.],
            estimated_gimbal_output=[2., -1.], max_gimbal_angle=8.))
        self.assertAlmostEqual(features['axis_error_degrees'], 90.)
        self.assertAlmostEqual(features['gyro_norm'], 5.)
        self.assertAlmostEqual(features['gimbal_fraction'], .25)

    def test_observed_two_hits_are_factual_and_replans_not_duplicate_rows(self):
        events = [decision(0, 1., 4, [4, 5]),
                  decision(0, 1.5, 4, [4, 5]), outcome(0, 2., "hit"),
                  decision(1, 2., 5, [5, 6]), outcome(1, 3., "hit")]
        rows = attempt_examples(events, 42)
        self.assertEqual(len(rows), 2)
        self.assertTrue(rows[0]["first_hit"])
        self.assertTrue(rows[0]["second_hit"])
        self.assertEqual(rows[0]["planned_departure_cosine"], 0.)
        self.assertIsNone(rows[1]["second_hit"])

    def test_unflown_or_changed_next_target_is_censored_not_failure(self):
        for first_outcome, next_target, reason in [
            ("miss", 5, "first_leg_missed"),
            ("hit", 6, "next_target_changed"),
        ]:
            with self.subTest(first_outcome=first_outcome, next_target=next_target):
                events = [decision(0, 1., 4, [4, 5]),
                          outcome(0, 2., first_outcome),
                          decision(1, 2., next_target, [next_target, 8]),
                          outcome(1, 3., "miss")]
                row = attempt_examples(events, 42)[0]
                self.assertIsNone(row["second_hit"])
                self.assertEqual(row["second_censor_reason"], reason)

    def test_replanned_second_target_censors_continuation(self):
        events = [decision(0, 1., 4, [4, 5]),
                  decision(0, 1.5, 4, [4, 6]), outcome(0, 2., "hit"),
                  decision(1, 2., 5, [5, 7]), outcome(1, 3., "hit")]
        rows = attempt_examples(events, 42)
        self.assertIsNone(rows[0]["second_hit"])
        self.assertEqual(rows[0]["second_censor_reason"],
                         "route_changed_during_first_leg")
        self.assertEqual(summarize(rows)["observed_second_legs"], 0)


if __name__ == "__main__":
    unittest.main()
