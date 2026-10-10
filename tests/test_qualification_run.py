import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import qualification_run as live
from scripts.qualification_seed_registry import SeedRegistry


class QualificationRunTests(unittest.TestCase):
    def test_seed_replacement_changes_only_scenario_random_seed(self):
        original = (b'scenario:\r\n  number: 4\r\n  random_seed: 0 # comment\r\n'
                    b'simulation:\r\n  time_step: 0.01\r\n')
        changed = live.replace_seed(original, 123456)
        self.assertEqual(changed, original.replace(b'random_seed: 0', b'random_seed: 123456'))
        with self.assertRaises(ValueError):
            live.replace_seed(original+original, 9)
        with self.assertRaises(ValueError):
            live.replace_seed(original, -1)

    def test_seed_file_is_restored_after_flight_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            scenario = folder/'scenario_4_parameters.yaml'
            original = b'scenario:\n  number: 4\n  random_seed: 0 # original\n'
            scenario.write_bytes(original)
            runtime = folder/'runtime'
            active = runtime/'active_seed_edit.json'
            with (patch.object(live, 'SCENARIO_FILE', scenario),
                  patch.object(live, 'RUNTIME', runtime), patch.object(live, 'ACTIVE', active)):
                with self.assertRaisesRegex(RuntimeError, 'flight failed'):
                    with live.temporary_seed(41827, 'test-run'):
                        self.assertIn(b'random_seed: 41827', scenario.read_bytes())
                        self.assertTrue(active.exists())
                        raise RuntimeError('flight failed')
                self.assertEqual(scenario.read_bytes(), original)
                self.assertFalse(active.exists())

    def test_recovery_restores_interrupted_edit_without_overwriting_other_change(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            scenario = folder/'scenario_4_parameters.yaml'
            original = b'scenario:\n  number: 4\n  random_seed: 0\n'
            scenario.write_bytes(original)
            runtime = folder/'runtime'
            active = runtime/'active_seed_edit.json'
            with (patch.object(live, 'SCENARIO_FILE', scenario),
                  patch.object(live, 'RUNTIME', runtime), patch.object(live, 'ACTIVE', active)):
                with self.assertRaisesRegex(RuntimeError, 'changed during run'):
                    with live.temporary_seed(42, 'test-run'):
                        scenario.write_bytes(b'scenario:\n  number: 4\n  random_seed: 43\n')
                self.assertTrue(active.exists())
                with self.assertRaisesRegex(RuntimeError, 'changed unexpectedly'):
                    live.recover_seed_edit()
                scenario.write_bytes(live.replace_seed(original, 42))
                live.recover_seed_edit()
                self.assertEqual(scenario.read_bytes(), original)
                self.assertFalse(active.exists())

    def test_submission_header_and_seed_registry_do_not_need_payload(self):
        with tempfile.TemporaryDirectory() as temporary:
            file = Path(temporary)/'sample_submission.json'
            file.write_text(json.dumps({'team': {'name': 'T', 'secret': 'DO_NOT_PRINT'},
                'leaderboard_info': {'random_seed': 123, 'final_reward': 7},
                'balloon_world_data': {'trajectories': [1]*2000}}), encoding='utf-8')
            self.assertEqual(live.parse_small_object_prefix(file, 'team')['name'], 'T')
            registry = SeedRegistry()
            registry.submission_header(file.read_bytes(), 'sample_submission.json')
            self.assertIn(123, registry.origins)
            self.assertNotIn('DO_NOT_PRINT', str(registry.as_dict()))


if __name__ == '__main__':
    unittest.main()
