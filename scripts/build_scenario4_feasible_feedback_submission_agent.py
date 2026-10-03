"""Build a standalone Scenario 4 feasible-feedback agent from reviewed source.

The output contains the existing standalone wind-profile planner and only the
adaptive gain function/class from the development module. It has no dependency
on development-agent imports, field data, seeds, or prior flights. Existing
files are never overwritten.
"""

import argparse
import ast
import copy
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AGENTS = ROOT / "BalloonPoppingGymEnv" / "agents"
BASE = AGENTS / "submission_scenario4_wind_profile_v1.py"
FEEDBACK = AGENTS / "scenario4_feasible_feedback_agent.py"
DEFAULT_OUTPUT = AGENTS / "submission_scenario4_feasible_feedback_v1.py"
OUTPUT_CLASS = "Scenario4FeasibleFeedbackSubmissionAgent"


def _unique_node(tree, kind, name):
    matches = [node for node in tree.body
               if isinstance(node, kind) and node.name == name]
    if len(matches) != 1:
        raise ValueError(f"Expected one {name} in source, found {len(matches)}")
    return copy.deepcopy(matches[0])


def build_source():
    base = BASE.read_text(encoding="utf-8")
    base_tree = ast.parse(base)
    _unique_node(base_tree, ast.ClassDef, "Scenario4SubmissionAgent")
    for name in ("sample_curve", "allocate_acceleration"):
        _unique_node(base_tree, ast.FunctionDef, name)
    if not any(isinstance(node, ast.Assign) and
               any(isinstance(target, ast.Name) and target.id == "G"
                   for target in node.targets) for node in base_tree.body):
        raise ValueError("Standalone base is missing gravity constant G")
    if any(isinstance(node, ast.ClassDef) and node.name == OUTPUT_CLASS
           for node in base_tree.body):
        raise ValueError("Standalone base already contains adaptive class")

    feedback_tree = ast.parse(FEEDBACK.read_text(encoding="utf-8"))
    gain = _unique_node(feedback_tree, ast.FunctionDef, "feasible_feedback_gain")
    agent = _unique_node(feedback_tree, ast.ClassDef,
                         "Scenario4FeasibleFeedbackAgent")
    if (len(agent.bases) != 1 or not isinstance(agent.bases[0], ast.Name) or
            agent.bases[0].id != "Scenario4WindProfileAgent"):
        raise ValueError("Unexpected feasible-feedback parent class")
    agent.name = OUTPUT_CLASS
    agent.bases = [ast.Name(id="Scenario4SubmissionAgent", ctx=ast.Load())]
    source = base.rstrip() + "\n\n\n" + ast.unparse(gain) + "\n\n\n" + ast.unparse(agent) + "\n"
    compile(source, "<scenario4-feasible-feedback-standalone>", "exec")
    return source


def write_source(output):
    output = Path(output)
    if output.suffix.lower() != ".py":
        raise ValueError("Standalone source output must be a .py file")
    source = build_source()
    # Exclusive creation also protects against an output appearing after a
    # preflight existence check, including the selected baseline standalone.
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(source)
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    digest = write_source(args.output)
    print(args.output)
    print("sha256=" + digest)


if __name__ == "__main__":
    main()
