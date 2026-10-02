"""Build a single-file, observation-only Scenario 4 wind-profile agent.

The official JSON packer embeds only the configured agent module's source.
This script appends our observed-wind estimator to the frozen standalone
time-allocation planner, without copying any balloon field, seed, simulator
state, or prerecorded action. Existing outputs are never overwritten.
"""

import argparse
import ast
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AGENTS = ROOT / "BalloonPoppingGymEnv" / "agents"
BASE = AGENTS / "submission_time_allocation_v1.py"
PROFILE = AGENTS / "scenario4_wind_profile_agent.py"


def build_source():
    base = BASE.read_text(encoding="utf-8")
    tree = ast.parse(PROFILE.read_text(encoding="utf-8"))
    selected = []
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        if node.name == "ObservedWindProfile":
            selected.append(ast.unparse(node))
        elif node.name == "Scenario4WindProfileAgent":
            node.name = "Scenario4SubmissionAgent"
            node.bases = [ast.Name(id="ChainSubmissionAgent", ctx=ast.Load())]
            selected.append(ast.unparse(node))
    if len(selected) != 2 or "class ChainSubmissionAgent(TimeAllocationAgent)" not in base:
        raise ValueError("Unexpected Scenario 4 source or standalone base")
    source = base.rstrip() + "\n\n\n" + "\n\n\n".join(selected) + "\n"
    compile(source, "<scenario4-standalone-agent>", "exec")
    return source


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path,
                        default=AGENTS / "submission_scenario4_wind_profile_v1.py")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing agent: {args.output}")
    source = build_source()
    args.output.write_text(source, encoding="utf-8", newline="\n")
    print(args.output)
    print("sha256=" + hashlib.sha256(source.encode("utf-8")).hexdigest())


if __name__ == "__main__":
    main()
