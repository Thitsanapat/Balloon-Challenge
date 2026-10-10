"""Reproducibly flatten the experimental adaptive entry policy for review.

This only builds a candidate source file. It does not change the qualification
launcher or certify score, parity, or rule compliance by itself.
"""

import argparse
import ast
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AGENTS = ROOT / "BalloonPoppingGymEnv/agents"
SOURCES = (
    ("submission_scenario4_pure_joint_v2.py",
     "9ff7f93d21a945d0479289be57bd893d8f33ab169fb1a6bfeb153ea442b6efb9"),
    ("scenario4_entry_horizon_agent.py",
     "344fe463533bf7801a1a96c0c99529b6449e32e459cf014c64e5a387df8f234f"),
    ("scenario4_adaptive_entry_agent.py",
     "11307fbbca6d9b389914c3b3abcd757ed699148f2e342837ee1f1f8eaab57509"),
)


def build_source():
    texts = []
    for filename, expected in SOURCES:
        raw = (AGENTS / filename).read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError(f"Source changed since experiment: {filename}")
        texts.append(raw.decode("utf-8"))
    output = texts[0].rstrip() + "\n"
    for source, expected_name in zip(
        texts[1:], ("Scenario4EntryHorizonAgent", "Scenario4AdaptiveEntryAgent")
    ):
        nodes = [node for node in ast.parse(source).body
                 if isinstance(node, ast.ClassDef) and node.name == expected_name]
        if len(nodes) != 1:
            raise ValueError(f"Expected exactly one {expected_name}")
        output += "\n" + ast.get_source_segment(source, nodes[0]) + "\n"
    compile(output, "<adaptive-entry-standalone>", "exec")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=AGENTS /
                        "submission_scenario4_adaptive_entry_v1.py")
    args = parser.parse_args()
    source = build_source()
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(source)
    print(args.output)
    print("sha256=" + hashlib.sha256(source.encode()).hexdigest())


if __name__ == "__main__":
    main()
