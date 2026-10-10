import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import yaml
from scripts.validate_multiscenario_pilot import configurations, validate_seeds, DEFAULT_SEEDS, HOLDOUT_SEEDS, run


class PilotValidationTests(unittest.TestCase):
    def test_reserved_seeds_are_unique_and_disjoint(self):
        validate_seeds(DEFAULT_SEEDS, [{'seed': 1234567}])
        for seeds in ([], [1000000007, 1000000007], [0], [1500000000]):
            with self.assertRaises(ValueError):
                validate_seeds(seeds, [])
        with self.assertRaisesRegex(ValueError, 'training'):
            validate_seeds(DEFAULT_SEEDS, [{'seed': DEFAULT_SEEDS[0]}])

    def test_same_frozen_controller_without_overrides(self):
        configs = configurations('trusted_model.zip')
        self.assertEqual(configs['baseline']['agent_kwargs'], {})
        self.assertEqual(set(configs['candidate']['agent_kwargs']), {'checkpoint'})
        self.assertTrue(all(c['scenario_number'] == 4 for c in configs.values()))
        self.assertTrue(all(c['leaderboard_submission'] is False for c in configs.values()))

    def test_holdout_pool_and_prior_plan_overlap(self):
        self.assertEqual(len(HOLDOUT_SEEDS), 10)
        validate_seeds(HOLDOUT_SEEDS, [{'seed': 1234567}], 'holdout', DEFAULT_SEEDS)
        with self.assertRaisesRegex(ValueError, 'holdout pool'):
            validate_seeds(DEFAULT_SEEDS, [], 'holdout')
        with self.assertRaisesRegex(ValueError, 'validation pool'):
            validate_seeds(HOLDOUT_SEEDS, [])
        with self.assertRaisesRegex(ValueError, 'prior plan'):
            validate_seeds(HOLDOUT_SEEDS, [], 'holdout', [HOLDOUT_SEEDS[0]])
        with self.assertRaisesRegex(ValueError, 'Unknown'):
            validate_seeds(HOLDOUT_SEEDS, [], 'train')

    def test_learned_parent_uses_same_adapter_without_controller_overrides(self):
        configs = configurations('new/final.zip', 'old/final.zip')
        self.assertEqual(configs['baseline']['agent_class_name'], configs['candidate']['agent_class_name'])
        self.assertEqual(configs['baseline']['agent_module_path'], configs['candidate']['agent_module_path'])
        self.assertEqual(set(configs['baseline']['agent_kwargs']), {'checkpoint'})
        self.assertNotEqual(configs['baseline']['agent_kwargs'], configs['candidate']['agent_kwargs'])

    def test_paired_runner_is_cpu_only_and_stops_on_failure(self):
        # Synthetic scores are fixtures, not flight results; keep test output quiet.
        self.enterContext(patch('builtins.print'))
        with TemporaryDirectory() as directory:
            root = Path(directory)
            pilot = root/'pilot'
            pilot.mkdir()
            checkpoint = pilot/'final_model.zip'
            checkpoint.write_bytes(b'trusted-test-placeholder')
            (pilot/'manifest.json').write_text(json.dumps({'source_sha256': {}}))
            (pilot/'episode_seeds.jsonl').write_text('{"seed": 1234567}\n')
            for name in ('scripts/evaluate_guidance.py', 'scripts/summarize_paired_evaluations.py',
                         'BalloonPoppingGymEnv/agents/multiscenario_ppo_policy_agent.py'):
                path = root/name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('# fixture\n')

            def flight(command, **kwargs):
                self.assertEqual(kwargs['env']['CUDA_VISIBLE_DEVICES'], '')
                self.assertEqual(kwargs['timeout'], 1200)
                config = yaml.safe_load(Path(command[4]).read_text())
                seed = int(command[command.index('--seeds')+1])
                candidate = config['agent_class_name'] == 'MultiScenarioPPOPolicyAgent'
                row = dict(scenario=4, seed=seed, config=config,
                           agent_sha256='candidate' if candidate else 'baseline',
                           agent_source_dependencies_sha256={},
                           agent_file_unchanged_during_run=True,
                           agent_sources_unchanged_during_run=True,
                           terminated=True, truncated=False, wall_seconds=1.,
                           score=2 if candidate else 1)
                Path(command[-1]).write_text(json.dumps([row]))

            prefix = 'scripts.validate_multiscenario_pilot.'
            learned_parent = root/'parent'
            learned_parent.mkdir()
            (learned_parent/'model.zip').write_bytes(b'distinct-trained-parent-fixture')
            (learned_parent/'manifest.json').write_text((pilot/'manifest.json').read_text())
            (learned_parent/'episode_seeds.jsonl').write_text('{"seed": 7654321}\n')
            with patch(prefix+'ROOT', root), patch(prefix+'subprocess.run', side_effect=flight) as child:
                run(checkpoint, root/'learned_pair', DEFAULT_SEEDS[:1],
                    baseline_checkpoint=learned_parent/'model.zip')
                self.assertEqual(child.call_count, 2)
                plan = json.loads((root/'learned_pair/plan.json').read_text())
                self.assertNotEqual(plan['baseline_checkpoint_sha256'], plan['checkpoint_sha256'])
                for label in ('baseline','candidate'):
                    self.assertEqual(plan['configurations'][label]['agent_class_name'], 'MultiScenarioPPOPolicyAgent')
                (learned_parent/'episode_seeds.jsonl').write_text(json.dumps({'seed':DEFAULT_SEEDS[0]})+'\n')
                with self.assertRaisesRegex(ValueError,'training'):
                    run(checkpoint, root/'parent_overlap', DEFAULT_SEEDS,
                        baseline_checkpoint=learned_parent/'model.zip')
            with patch(prefix+'ROOT', root):
                with self.assertRaisesRegex(ValueError, 'distinct'):
                    run(checkpoint, root/'identical', DEFAULT_SEEDS, baseline_checkpoint=checkpoint)
            with patch(prefix+'ROOT', root), patch(prefix+'subprocess.run', side_effect=flight) as child:
                run(checkpoint, root/'success', DEFAULT_SEEDS)
                self.assertEqual(child.call_count, 6)
                summary = json.loads((root/'success/summary.json').read_text())
                self.assertEqual(summary['mean_score_delta'], 1.)
                self.assertTrue(summary['frozen_sources_and_model_unchanged'])
                with self.assertRaises(FileExistsError):
                    run(checkpoint, root/'success', DEFAULT_SEEDS)
            prior = root/'success/plan.json'
            with patch(prefix+'ROOT', root), patch(prefix+'subprocess.run', side_effect=flight) as child:
                with self.assertRaisesRegex(ValueError, 'requires prior'):
                    run(checkpoint, root/'holdout', HOLDOUT_SEEDS, 'holdout')
                child.assert_not_called()
                run(checkpoint, root/'holdout', HOLDOUT_SEEDS, 'holdout', [prior])
                self.assertEqual(child.call_count, 20)
                summary = json.loads((root/'holdout/summary.json').read_text())
                self.assertEqual(summary['split'], 'holdout')
                self.assertEqual(summary['complete_pair_count'], 10)
                with self.assertRaisesRegex(ValueError, 'prior plan'):
                    run(checkpoint, root/'reuse', HOLDOUT_SEEDS, 'holdout', [root/'holdout/plan.json'])
            with patch(prefix+'ROOT', root), patch(prefix+'subprocess.run',
                    side_effect=subprocess.CalledProcessError(1, 'fixture')) as child:
                with self.assertRaises(subprocess.CalledProcessError):
                    run(checkpoint, root/'failed', DEFAULT_SEEDS)
                self.assertEqual(child.call_count, 1)
                status = json.loads((root/'failed/status.json').read_text())
                self.assertEqual(status['status'], 'failed')
                self.assertEqual(status['completed'], [])
                self.assertFalse((root/'failed/summary.json').exists())


if __name__ == '__main__':
    unittest.main()
