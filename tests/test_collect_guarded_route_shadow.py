import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.collect_guarded_route_shadow import (
    DEFAULT_CONFIG, source_manifest, validate_training_seeds,
)


class GuardedRouteCollectionTests(unittest.TestCase):
    def test_selected_agent_source_is_frozen_in_manifest(self):
        config, manifest = source_manifest(DEFAULT_CONFIG)
        self.assertEqual(config['agent_class_name'],
                         'Scenario4GuardedInnovationAgent')
        self.assertTrue(any(Path(path).name ==
                            'submission_scenario4_guarded_innovation_v1.py'
                            for path in manifest))

    def test_seed_pool_rejects_reuse_and_duplicates(self):
        with tempfile.TemporaryDirectory() as temp:
            registry = Path(temp) / 'used_seeds.json'
            registry.write_text(json.dumps({'seeds': {'123': {}}}),
                                encoding='utf-8')
            with patch('scripts.collect_guarded_route_shadow.SEED_REGISTRY',
                       registry):
                validate_training_seeds([456])
                with self.assertRaisesRegex(ValueError, 'already recorded'):
                    validate_training_seeds([123])
                with self.assertRaises(ValueError):
                    validate_training_seeds([456, 456])


if __name__ == '__main__':
    unittest.main()
