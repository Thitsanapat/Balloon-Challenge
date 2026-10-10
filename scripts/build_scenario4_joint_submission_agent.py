"""Bundle joint repair with the current planner dispatch hooks.

Output is a separate review artifact, never the live module. No seed-specific
information, telemetry or model-training labels are included.
"""

import argparse
import ast
import hashlib
from pathlib import Path

from scripts.build_scenario4_guarded_submission_agent import build_source as guarded_source


ROOT = Path(__file__).resolve().parents[1]
AGENTS = ROOT / 'BalloonPoppingGymEnv/agents'


def build_source():
    base = guarded_source()
    if base != (AGENTS / 'submission_scenario4_guarded_innovation_v1.py').read_text():
        raise ValueError('Reviewed guarded standalone changed')
    # The old reviewed standalone predates committed-leg recovery hooks.
    # Merely appending an override compiles, but the old planner never calls
    # it. Replace these base-class definitions BEFORE their subclasses are
    # created, preserving the live reviewed source file byte-for-byte.
    base_tree = ast.parse(base)
    replacements = {}
    for filename, class_name in (
            ('chain_beam_agent.py', 'ChainBeamAgent'),
            ('chain_launch_agent.py', 'ChainLaunchAgent')):
        parsed = ast.parse((AGENTS / filename).read_text(encoding='utf-8'))
        replacements[class_name] = next(
            node for node in parsed.body
            if isinstance(node, ast.ClassDef) and node.name == class_name)
    replaced = set()
    for index, node in enumerate(base_tree.body):
        if isinstance(node, ast.ClassDef) and node.name in replacements:
            base_tree.body[index] = replacements[node.name]
            replaced.add(node.name)
    if replaced != set(replacements):
        raise ValueError('Expected planner classes missing from standalone')
    tree = ast.parse((AGENTS / 'scenario4_pure_joint_refresh_agent.py').read_text())
    definitions = [node for node in tree.body if isinstance(node, ast.ClassDef)
                   and node.name == 'Scenario4PureJointRefreshAgent']
    if len(definitions) != 1:
        raise ValueError('Expected one pure joint agent definition')
    base_tree.body.append(definitions[0])
    source = ast.unparse(base_tree) + '\n'
    compile(source, '<scenario4-pure-joint-standalone>', 'exec')
    return source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=(
        ROOT / 'review/scenario4_pure_joint_candidate_v2.py'))
    args = parser.parse_args()
    source = build_source()
    # Exclusive create: a reviewed artifact must not silently change.
    with args.output.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(source)
    print(args.output)
    print('sha256=' + hashlib.sha256(source.encode()).hexdigest())


if __name__ == '__main__':
    main()
