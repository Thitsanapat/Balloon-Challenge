import unittest

from scripts.fit_route_continuation import transfer


def row(seed, index):
    return dict(seed=seed, planned_route_length=2 + index % 3,
                planned_first_seconds=2. + index % 4,
                planned_second_seconds=1. + index % 2,
                planned_departure_cosine=None if index % 5 == 0 else .5,
                first_hit=index % 2 == 0,
                second_hit=index % 3 == 0)


class ContinuationFitTests(unittest.TestCase):
    def test_disjoint_transfer_has_finite_scores(self):
        train = [row(1 + i // 5, i) for i in range(25)]
        test = [row(10 + i // 5, i) for i in range(25)]
        result = transfer(train, test, 'second_hit')
        self.assertEqual(result['test_labels'], 25)
        self.assertTrue(0 <= result['learned']['brier'] <= 1)

    def test_overlapping_fields_rejected(self):
        rows = [row(1, i) for i in range(25)]
        with self.assertRaisesRegex(ValueError, 'disjoint'):
            transfer(rows, rows, 'first_hit')


if __name__ == '__main__':
    unittest.main()
