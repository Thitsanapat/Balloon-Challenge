"""Check the adaptive standalone source before any official evaluation."""

import ast
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from BalloonPoppingGymEnv.agents.scenario4_feasible_feedback_agent import (
    feasible_feedback_gain,
)
from scripts.build_scenario4_feasible_feedback_submission_agent import (
    BASE,
    FEEDBACK,
    OUTPUT_CLASS,
    build_source,
    write_source,
)


def named_node(tree, kind, name):
    return next(node for node in tree.body
                if isinstance(node, kind) and node.name == name)


class FeasibleFeedbackSubmissionBundleTests(unittest.TestCase):
    def test_source_keeps_frozen_base_and_exact_gain_logic(self):
        base = BASE.read_text(encoding="utf-8")
        generated = build_source()
        self.assertTrue(generated.startswith(base.rstrip() + "\n\n\n"))
        generated_tree = ast.parse(generated)
        base_tree = ast.parse(base)
        feedback_tree = ast.parse(FEEDBACK.read_text(encoding="utf-8"))
        generated_imports = [ast.dump(node) for node in generated_tree.body
                             if isinstance(node, (ast.Import, ast.ImportFrom))]
        base_imports = [ast.dump(node) for node in base_tree.body
                        if isinstance(node, (ast.Import, ast.ImportFrom))]
        self.assertEqual(generated_imports, base_imports)
        self.assertEqual(
            ast.dump(named_node(generated_tree, ast.FunctionDef,
                                "feasible_feedback_gain")),
            ast.dump(named_node(feedback_tree, ast.FunctionDef,
                                "feasible_feedback_gain")),
        )
        original_agent = named_node(feedback_tree, ast.ClassDef,
                                    "Scenario4FeasibleFeedbackAgent")
        generated_agent = named_node(generated_tree, ast.ClassDef, OUTPUT_CLASS)
        self.assertEqual([ast.dump(node) for node in generated_agent.body],
                         [ast.dump(node) for node in original_agent.body])
        self.assertEqual(generated_agent.bases[0].id, "Scenario4SubmissionAgent")

    def test_generated_gain_matches_modular_gain_on_feasible_and_saturated_inputs(self):
        namespace = {"__name__": "scenario4_feasible_feedback_generated_test"}
        exec(compile(build_source(), "<generated-agent>", "exec"), namespace)
        generated_gain = namespace["feasible_feedback_gain"]
        zero = np.zeros(3)
        cases = (
            (zero, zero, zero, zero, 15.0, 1.0),
            (zero, np.array([100., 0., 0.]), zero, zero, 15.0, 1.0),
            (np.array([2., -1., 0.]), np.array([.5, 1., -.2]),
             np.array([-.3, .2, .1]), np.array([.1, -.2, 0.]), 12.0, .7),
        )
        gains = np.linspace(.7, 1.4, 9)
        for case in cases:
            with self.subTest(case=case):
                self.assertEqual(generated_gain(*case, gains),
                                 feasible_feedback_gain(*case, gains))

    def test_writer_creates_new_python_file_once(self):
        with TemporaryDirectory() as directory:
            destination = Path(directory) / "new_standalone.py"
            digest = write_source(destination)
            self.assertEqual(destination.read_text(encoding="utf-8"), build_source())
            self.assertEqual(len(digest), 64)
            with self.assertRaises(FileExistsError):
                write_source(destination)
            with self.assertRaises(ValueError):
                write_source(Path(directory) / "result.json")


if __name__ == "__main__":
    unittest.main()
