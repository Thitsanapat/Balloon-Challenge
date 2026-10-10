import ast
import unittest

from scripts.build_scenario4_guarded_submission_agent import (
    AGENTS,
    build_source,
)


class GuardedSubmissionBuildTests(unittest.TestCase):
    def test_standalone_matches_reproducible_builder(self):
        target = AGENTS / "submission_scenario4_guarded_innovation_v1.py"
        self.assertEqual(target.read_text(encoding="utf-8"), build_source())

    def test_no_runtime_imports_from_experimental_agent_modules(self):
        tree = ast.parse(build_source())
        modules = [node.module for node in tree.body
                   if isinstance(node, ast.ImportFrom)]
        self.assertEqual(modules,
                         ["functools", "scipy.optimize",
                          "BalloonPoppingGymEnv.agents.base_agent"])
        classes = {node.name: node for node in tree.body
                   if isinstance(node, ast.ClassDef)}
        self.assertIn("Scenario4GuardedInnovationAgent", classes)
        self.assertIn("Scenario4ForecastInnovationAgent", classes)


if __name__ == "__main__":
    unittest.main()
