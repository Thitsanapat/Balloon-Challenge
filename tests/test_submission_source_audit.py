import unittest
from scripts.audit_agent_submission import source_review


class SourceReviewTests(unittest.TestCase):
    def test_io_and_private_seed_are_flagged(self):
        imports, flags = source_review("import os\nx = open('field.json')\ny = params['random_seed']\n")
        self.assertIn('os', imports)
        self.assertTrue(any('open' in flag for flag in flags))
        self.assertTrue(any('seed' in flag for flag in flags))

    def test_observation_math_is_not_flagged(self):
        _, flags = source_review("import numpy as np\nx = np.asarray(observation['balloon_states']).copy()\n")
        self.assertEqual(flags, [])


if __name__=='__main__':
    unittest.main()
