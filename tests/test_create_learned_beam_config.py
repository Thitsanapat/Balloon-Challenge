import json
import tempfile
import unittest
from pathlib import Path

import yaml

from scripts.create_learned_beam_config import main


class LearnedBeamConfigTests(unittest.TestCase):
    def test_writes_numeric_self_contained_ranker_config(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root/'model.json'
            model.write_text(json.dumps({key: [1.]*16 for key in (
                'coefficients', 'feature_mean', 'feature_scale')}), encoding='utf-8')
            template = root/'template.yaml'
            template.write_text('agent_kwargs: {}\nleaderboard_submission: true\n', encoding='utf-8')
            output = root/'out.yaml'
            import sys
            old = sys.argv
            try:
                sys.argv = ['tool', '--model', str(model), '--template', str(template),
                            '--output', str(output), '--agent-module', 'agent.py',
                            '--rank-bonus', '.2']
                main()
            finally:
                sys.argv = old
            config = yaml.safe_load(output.read_text(encoding='utf-8'))
            self.assertEqual(config['agent_class_name'], 'LearnedBeamSubmissionAgent')
            self.assertEqual(config['agent_kwargs']['rank_bonus'], .2)
            self.assertFalse(config['leaderboard_submission'])


if __name__ == '__main__':
    unittest.main()
