"""Build a standalone, observation-only guarded Scenario-4 agent.

The official submission JSON embeds only its configured agent module. This
builder combines the reviewed standalone wind-profile controller with the
small endpoint-innovation and safety-gate additions. It does not copy seed,
balloon-field, replay, or hidden simulator data. Existing output is never
overwritten.
"""

import argparse
import ast
import hashlib
from pathlib import Path

from scripts.build_scenario4_submission_agent import build_source as profile_source


ROOT = Path(__file__).resolve().parents[1]
AGENTS = ROOT / "BalloonPoppingGymEnv" / "agents"
PROFILE_STANDALONE = AGENTS / "submission_scenario4_wind_profile_v1.py"
INNOVATION = AGENTS / "scenario4_forecast_innovation_agent.py"
GUARDED = AGENTS / "scenario4_guarded_innovation_agent.py"


class StandaloneNames(ast.NodeTransformer):
    def visit_Name(self, node):
        if node.id == "Scenario4WindProfileAgent":
            return ast.copy_location(ast.Name(id="Scenario4SubmissionAgent",
                                              ctx=node.ctx), node)
        return node


def definitions(path, names):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found = {node.name: node for node in tree.body
             if isinstance(node, (ast.FunctionDef, ast.ClassDef))
             and node.name in names}
    if set(found) != set(names):
        raise ValueError(f"Expected definitions missing from {path.name}")
    translator = StandaloneNames()
    return [ast.unparse(translator.visit(found[name])) for name in names]


def build_source():
    base = profile_source()
    if PROFILE_STANDALONE.read_text(encoding="utf-8") != base:
        raise RuntimeError("Reviewed standalone wind-profile source changed")
    parts = definitions(INNOVATION, (
        "smooth_innovation_gate", "observed_endpoint_innovation",
        "bounded_innovation_force", "Scenario4ForecastInnovationAgent"))
    parts += definitions(GUARDED, (
        "accept_terminal_correction", "Scenario4GuardedInnovationAgent"))
    source = base.rstrip() + "\n\n\n" + "\n\n\n".join(parts) + "\n"
    compile(source, "<scenario4-guarded-standalone>", "exec")
    if "from BalloonPoppingGymEnv.agents.scenario4_" in source:
        raise RuntimeError("Standalone still imports an experimental agent")
    return source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=(
        AGENTS / "submission_scenario4_guarded_innovation_v1.py"))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing agent: {args.output}")
    source = build_source()
    args.output.write_text(source, encoding="utf-8", newline="\n")
    print(args.output)
    print("sha256=" + hashlib.sha256(source.encode("utf-8")).hexdigest())


if __name__ == "__main__":
    main()
