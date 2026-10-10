import unittest

from scripts.fit_combined_guarded_route import calibration_gate


class CombinedGuardedRouteFitTests(unittest.TestCase):
    def report(self, brier=0.24, log_loss=0.67):
        return dict(fields=24, attempts=145,
                    feature_names=(
                        'log_time_to_deadline', 'relative_range',
                        'closing_speed', 'required_lateral_fraction',
                        'required_total_fraction', 'available_acceleration'),
                    prior=dict(brier=0.25, log_loss=0.69),
                    learned=dict(brier=brier, log_loss=log_loss))

    def test_requires_both_predeclared_improvements(self):
        self.assertTrue(calibration_gate(self.report()))
        self.assertFalse(calibration_gate(self.report(brier=0.248)))
        self.assertFalse(calibration_gate(self.report(log_loss=0.686)))

    def test_rejects_small_or_wrong_schema(self):
        candidate = self.report()
        candidate['attempts'] = 99
        self.assertFalse(calibration_gate(candidate))
        candidate = self.report()
        candidate['feature_names'] = ()
        self.assertFalse(calibration_gate(candidate))


if __name__ == '__main__':
    unittest.main()
