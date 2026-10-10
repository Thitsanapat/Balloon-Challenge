import json
from pathlib import Path
import tempfile
import unittest

from BalloonPoppingGymEnv.agents.scenario4_learned_route_agent import FEATURE_NAMES
from scripts.package_guarded_capture_model import TRAINING_SEEDS, package


class PackageGuardedCaptureTests(unittest.TestCase):
    def report(self):
        return dict(schema='route_capture_ml_group_holdout_v1',
                    fields=12, attempts=48, feature_names=list(FEATURE_NAMES),
                    folds=[dict(seed=seed) for seed in TRAINING_SEEDS],
                    prior=dict(brier=.25, log_loss=.70),
                    learned=dict(brier=.22, log_loss=.65),
                    full_fit_prototype=dict(center=[0.] * 6,
                                            scale=[1.] * 6,
                                            coefficients=[0.] * 7),
                    source_archive_sha256='a' * 64)

    def test_weights_strip_training_seed_ids(self):
        with tempfile.TemporaryDirectory() as temp:
            source, output = Path(temp) / 'report.json', Path(temp) / 'weights.json'
            source.write_text(json.dumps(self.report()), encoding='utf-8')
            payload = package(source, output)
            self.assertTrue(output.is_file())
            self.assertNotIn('folds', payload)
            self.assertNotIn('seeds', payload)
            self.assertNotIn(str(TRAINING_SEEDS[0]), output.read_text())

    def test_calibration_gate_rejects_worse_model(self):
        report = self.report()
        report['learned']['brier'] = .26
        with tempfile.TemporaryDirectory() as temp:
            source, output = Path(temp) / 'report.json', Path(temp) / 'weights.json'
            source.write_text(json.dumps(report), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'did not pass'):
                package(source, output)
            self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()
