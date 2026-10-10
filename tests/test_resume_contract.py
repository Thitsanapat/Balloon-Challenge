import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from BalloonPoppingGymEnv.training.resume_contract import sha256, validate_parent


class ResumeContractTests(unittest.TestCase):
    def test_exact_parent_only(self):
        with TemporaryDirectory() as d:
            root = Path(d)
            checkpoint = root/'model.zip'
            checkpoint.write_bytes(b'fixture')
            source = root/'controller.py'
            source.write_text('# fixture')
            manifest = dict(feature_schema='test', mixture=dict(scenarios=[4],probabilities=[1.]),
                            settings=dict(n_steps=128,batch_size=128),source_sha256={'controller.py':sha256(source)})
            (root/'manifest.json').write_text(json.dumps(manifest))
            args = [checkpoint,sha256(checkpoint),root,'test',[4],[1.],128,128]
            _, provenance = validate_parent(*args)
            self.assertEqual(provenance['checkpoint_sha256'],sha256(checkpoint))
            for index, value, message in [(1,'bad','hash'),(3,'old','schema'),(4,[3],'mixture'),(6,64,'rollout')]:
                bad = args.copy()
                bad[index] = value
                with self.assertRaisesRegex(ValueError,message):
                    validate_parent(*bad)
            source.write_text('# changed')
            with self.assertRaisesRegex(ValueError,'source differs'):
                validate_parent(*args)


if __name__ == '__main__':
    unittest.main()
