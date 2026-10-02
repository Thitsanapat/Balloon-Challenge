"""Confirm the Scenario 4 JSON agent can be reproduced from source only."""

import ast
import unittest

import numpy as np

from scripts.build_scenario4_submission_agent import build_source
from BalloonPoppingGymEnv.agents.scenario4_wind_profile_agent import ObservedWindProfile


class Scenario4SubmissionBundleTests(unittest.TestCase):
    def test_bundle_has_no_development_agent_imports(self):
        source = build_source()
        tree = ast.parse(source)
        imports = [node.module for node in tree.body
                   if isinstance(node, ast.ImportFrom)]
        self.assertEqual(imports, ["functools", "scipy.optimize",
                                   "BalloonPoppingGymEnv.agents.base_agent"])
        self.assertIn("class Scenario4SubmissionAgent(ChainSubmissionAgent):", source)

    def test_profile_math_matches_modular_source(self):
        scope = {"__name__": "scenario4_generated_test"}
        exec(compile(build_source(), "<generated-agent>", "exec"), scope)
        generated = scope["ObservedWindProfile"]()
        modular = ObservedWindProfile()
        states = np.zeros((9, 6))
        states[:, 2] = np.arange(80., 125., 5.)
        states[:, 3] = 2. + .1 * (states[:, 2] - 100.)
        states[:, 5] = 5.
        status = np.ones(9, dtype=int)
        np.testing.assert_allclose(generated.correction(states, status, 4, 2.),
                                   modular.correction(states, status, 4, 2.))


if __name__ == "__main__":
    unittest.main()
