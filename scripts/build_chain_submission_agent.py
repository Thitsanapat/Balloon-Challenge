"""Bundle the observation-only chain planner and controller for reproduction."""

import ast
import argparse
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AGENTS = ROOT/'BalloonPoppingGymEnv/agents'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--with-launch', action='store_true')
    parser.add_argument('--final-approach', action='store_true')
    parser.add_argument('--projected-tracking', action='store_true')
    parser.add_argument('--constrained', action='store_true')
    parser.add_argument('--time-allocation', action='store_true')
    parser.add_argument('--learned-beam', action='store_true',
                        help='Append the compact observation-only beam ranker')
    parser.add_argument('--output', type=Path, help='New agent file; existing files are never overwritten')
    args = parser.parse_args()
    if args.learned_beam:
        args.time_allocation = True
    if args.time_allocation and (args.projected_tracking or args.constrained):
        parser.error('Time allocation cannot be combined with the other experimental bundles')
    final_approach = args.final_approach or args.projected_tracking or args.constrained or args.time_allocation
    with_launch = args.with_launch or final_approach
    if final_approach and args.output is None:
        parser.error('New variants require an explicit --output path')
    output = args.output or AGENTS/('submission_chain_launch_v1.py' if with_launch else 'submission_chain_v1.py')
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite existing agent: {output}')
    parts = ['''"""Observation-only joint-derivative chain guidance.

Generated from controller and online planner source by
scripts/build_chain_submission_agent.py. No training data, stored routes,
seed lookups, or simulator internals are used.
"""
from functools import lru_cache
import numpy as np
from scipy.optimize import minimize
from BalloonPoppingGymEnv.agents.base_agent import BaseAgent
''']
    filenames = ['submission_sqp_v2.py','momentum_beam_agent.py','chain_beam_agent.py']
    selected = {}
    if with_launch:
        parts.append('import copy')
        filenames.append('chain_launch_agent.py')
    if args.constrained:
        filenames.extend(['constrained_chain_agent.py', 'constrained_beam_agent.py'])
        selected['constrained_chain_agent.py'] = {'constraint_margins', 'repair_margins', 'repair_derivatives'}
    if final_approach:
        filenames.append('final_approach_agent.py')
        selected['final_approach_agent.py'] = {'FinalApproachMixin', 'FinalApproachAgent'}
        if args.constrained:
            selected['final_approach_agent.py'].add('FinalApproachConstrainedAgent')
    if args.projected_tracking:
        filenames.append('projected_tracking_agent.py')
        selected['projected_tracking_agent.py'] = {'project_thrust', 'ProjectedTrackingMixin', 'ProjectedTrackingAgent'}
        if args.constrained:
            selected['projected_tracking_agent.py'].add('ProjectedConstrainedAgent')
    if args.time_allocation:
        filenames.extend(['constrained_chain_agent.py', 'opportunistic_chain_agent.py', 'time_allocation_agent.py'])
        selected['constrained_chain_agent.py'] = {'constraint_margins', 'repair_margins'}
        selected['opportunistic_chain_agent.py'] = {'closest_approaches'}
    for filename in filenames:
        tree = ast.parse((AGENTS/filename).read_text(encoding='utf-8'))
        for node in tree.body:
            if filename in selected and getattr(node, 'name', None) not in selected[filename]:
                continue
            if isinstance(node,(ast.Import,ast.ImportFrom)):
                continue
            if isinstance(node,ast.Expr) and isinstance(node.value,ast.Constant):
                continue
            parts.append(ast.unparse(node))
    parent = 'ChainLaunchAgent' if with_launch else 'ChainBeamAgent'
    if final_approach:
        parent = 'FinalApproachConstrainedAgent' if args.constrained else 'FinalApproachAgent'
    if args.projected_tracking:
        parent = 'ProjectedConstrainedAgent' if args.constrained else 'ProjectedTrackingAgent'
    if args.time_allocation:
        parent = 'TimeAllocationAgent'
    parts.append(f'class ChainSubmissionAgent({parent}):\n    pass\n')
    if args.learned_beam:
        # The submission contains the small fixed numeric ranker, not an import
        # of development code or a learned-weight file.  Only current released
        # observation features are appended to the self-contained bundle.
        for filename, names in (
            ('route_selector_agent.py', {'SELECTOR_FEATURES', 'ranked_candidate_ids',
                                         'selector_features'}),
            ('learned_beam_agent.py', {'RANK_FEATURES', 'LearnedBeamAgent'}),
        ):
            tree = ast.parse((AGENTS/filename).read_text(encoding='utf-8'))
            for node in tree.body:
                node_name = getattr(node, 'name', None)
                if (node_name is None and isinstance(node, ast.Assign)
                        and len(node.targets) == 1
                        and isinstance(node.targets[0], ast.Name)):
                    node_name = node.targets[0].id
                if node_name not in names:
                    continue
                source = ast.unparse(node)
                if node_name == 'LearnedBeamAgent':
                    source = source.replace(
                        'class LearnedBeamAgent(ChainSubmissionAgent):',
                        'class LearnedBeamSubmissionAgent(ChainSubmissionAgent):',
                        1,
                    )
                parts.append(source)
    source = '\n\n'.join(parts)+'\n'
    compile(source,str(output),'exec')
    output.write_text(source,encoding='utf-8',newline='\n')
    print(output)
    print('sha256='+hashlib.sha256(source.encode()).hexdigest())


if __name__=='__main__':
    main()
