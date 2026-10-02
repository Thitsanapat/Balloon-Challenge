"""Fast wiring checks; these tests never launch a simulator or train PPO."""

import hashlib
import io
from pathlib import Path
import unittest
from contextlib import redirect_stderr
from unittest.mock import patch

from scripts import train_official_residual_ppo as trainer


class OfficialPPOScenarioTests(unittest.TestCase):
    def test_scenario_one_remains_default_and_four_is_selectable(self):
        parser = trainer.build_parser()
        self.assertEqual(parser.parse_args(['--output', 'unused']).scenario, 1)
        self.assertEqual(parser.parse_args(['--output', 'unused', '--scenario', '4']).scenario, 4)
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parser.parse_args(['--output', 'unused', '--scenario', '5'])

    def test_scenario_four_reaches_official_training_wrapper(self):
        reset_log = Path('unused_reset_log.jsonl')
        with patch.object(trainer, 'OfficialResidualEnv') as wrapper:
            result = trainer.make_training_env(4, reset_log)
        wrapper.assert_called_once_with(scenario=4, reset_log=reset_log)
        self.assertIs(result, wrapper.return_value)

    def test_selected_parameter_files_are_hashed(self):
        scenario_hashes = trainer.hashes(4)
        for suffix in ('parameters', 'given_parameters'):
            relative = (f'BalloonPoppingGymEnv/envs/scenario_parameters/'
                        f'scenario_4_{suffix}.yaml')
            expected = hashlib.sha256((trainer.ROOT / relative).read_bytes()).hexdigest()
            self.assertEqual(scenario_hashes[relative], expected)
        self.assertNotIn('BalloonPoppingGymEnv/envs/scenario_parameters/scenario_1_parameters.yaml',
                         scenario_hashes)


if __name__ == '__main__':
    unittest.main()
