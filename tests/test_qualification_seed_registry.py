import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
import zipfile

import numpy as np

from scripts.qualification_seed_registry import collect


class QualificationSeedRegistryTests(unittest.TestCase):
    def test_reads_submissions_plans_and_conservative_legacy_rng_prefix(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            results = root/'BalloonPoppingGymEnv/evaluation/results'
            results.mkdir(parents=True)
            scenario = root/'BalloonPoppingGymEnv/envs/scenario_parameters/scenario_4_parameters.yaml'
            scenario.parent.mkdir(parents=True)
            scenario.write_text('scenario:\n  number: 4\n  random_seed: 0\n', encoding='utf-8')
            (results/'old_submission.json').write_text(json.dumps({
                'team': {'name': 'T', 'secret': 'NEVER_COPY_ME'},
                'leaderboard_info': {'random_seed': 17}}), encoding='utf-8')
            with zipfile.ZipFile(results/'plans.zip', 'w') as archive:
                archive.writestr('data/plan.json', json.dumps({'seeds': [31, 32]}))
            with tarfile.open(results/'old_training.tgz', 'w:gz') as archive:
                entries = {
                    'pilot/manifest.json': json.dumps({'settings': {'seed': 2026, 'envs': 1}}),
                    'pilot/episodes.jsonl': '{"score": 4}\n',
                    'pilot/worker_0.monitor.csv': '#{}\nr,l,t\n4,200,1.0\n',
                }
                for name, value in entries.items():
                    raw = value.encode()
                    info = tarfile.TarInfo(name)
                    info.size = len(raw)
                    archive.addfile(info, io.BytesIO(raw))
            registry = collect(root)
            expected_legacy = int(np.random.default_rng(2026).integers(100_000, 2**31-1))
            self.assertTrue({17, 31, 32, expected_legacy} <= set(registry.origins))
            self.assertNotIn('NEVER_COPY_ME', str(registry.as_dict()))
            self.assertTrue(any('exact episodes unverified' in warning
                                for warning in registry.warnings))


if __name__ == '__main__':
    unittest.main()
