"""Credential-safe evidence report, not an organizer rules certification.

Reads the agent embedded in a submission rather than assuming the current local
agent is what was submitted. Never executes that embedded source or prints team
credentials. Optionally runs the unchanged official data verifier.
"""

import argparse
import ast
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def source_review(source):
    tree = ast.parse(source)
    imports, flags = [], []
    allowed = {'numpy','scipy.optimize','functools','copy',
               'BalloonPoppingGymEnv.agents.base_agent'}
    forbidden = {'open','eval','exec','__import__','load','loads','loadtxt',
                 'read_text','read_bytes','read_csv','fromfile','getattr','setattr'}
    for node in ast.walk(tree):
        if isinstance(node,(ast.Import,ast.ImportFrom)):
            modules = [node.module] if isinstance(node,ast.ImportFrom) else [a.name for a in node.names]
            imports.extend(modules)
            for module in modules:
                if module not in allowed:
                    flags.append(f'import requiring manual review: {module}')
        if isinstance(node,ast.Call):
            name = node.func.id if isinstance(node.func,ast.Name) else node.func.attr if isinstance(node.func,ast.Attribute) else ''
            if name in forbidden:
                flags.append(f'call requiring manual review: {name}, line {node.lineno}')
        if isinstance(node,ast.Constant) and isinstance(node.value,str):
            if node.value in {'random_seed','seed_number','balloon_trajectory','_balloon_flights','_rocket_flight'}:
                flags.append(f'private-data/seed key, line {node.lineno}')
        if isinstance(node,ast.Attribute) and node.attr in {'_balloon_flights','_rocket_flight','_balloon_states','np_random_seed'}:
            flags.append(f'private simulator attribute, line {node.lineno}')
    return sorted(set(imports)), sorted(set(flags))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('submission',type=Path)
    parser.add_argument('--agent',type=Path,required=True)
    parser.add_argument('--report',type=Path,required=True)
    parser.add_argument('--verify',action='store_true')
    parser.add_argument('--official-reference',default='bd9bb51',
                        help='Official release commit or tag used to check simulator files')
    args = parser.parse_args()
    raw = args.submission.read_bytes()
    payload = json.loads(raw)
    if not isinstance(payload,dict):
        raise ValueError('Submission must be a JSON object')
    source = payload['agent_info']['agent_module_file']
    imports,flags = source_review(source)
    files = ['BalloonPoppingGymEnv/envs', 'BalloonPoppingGymEnv/evaluation/evaluate.py',
             'BalloonPoppingGymEnv/evaluation/results/utils.py','scripts/verify_submission.py']
    check = subprocess.run(['git','diff','--exit-code',args.official_reference,'--',*files],
                           cwd=ROOT,capture_output=True,text=True)
    info = payload['leaderboard_info']
    result = {
        'scope':'Evidence checks only; organizer retains final rules judgment. Static scans do not prove absence of all hidden behavior.',
        'submission_file':args.submission.name,
        'submission_sha256':hashlib.sha256(raw).hexdigest(),
        'json_object':True, 'format_version':payload.get('format_version'),
        'scenario':info['scenario_number'],'score':info['final_reward'],
        'embedded_source_sha256':hashlib.sha256(source.encode()).hexdigest(),
        'embedded_source_matches_local':source==args.agent.read_text(encoding='utf-8'),
        'imports':imports,'source_review_flags':flags,
        f'official_components_unchanged_vs_{args.official_reference}':check.returncode==0,
        'rules_reference':'https://github.com/ARRC-Rocket/BalloonPoppingChallenge/discussions/165',
    }
    if args.verify:
        checked = subprocess.run([sys.executable,str(ROOT/'scripts/verify_submission.py'),
                                  str(args.submission.resolve())],cwd=ROOT,capture_output=True,text=True)
        result['official_verifier_exit_code'] = checked.returncode
        result['official_verifier_findings'] = [line.strip() for line in checked.stdout.splitlines()
                                                if '[ok  ]' in line or '[FAIL]' in line]
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))
    return int(bool(flags) or not result['embedded_source_matches_local']
               or check.returncode!=0 or result.get('official_verifier_exit_code',0)!=0)


if __name__=='__main__':
    sys.exit(main())
