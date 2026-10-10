"""Package the parity-checked joint candidate and reproducible source inputs."""

import hashlib
from importlib import metadata
import json
from pathlib import Path
import platform
import re
from zipfile import ZIP_DEFLATED, ZipFile

import yaml

from scripts.build_scenario4_joint_submission_agent import build_source


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'review/scenario4_pure_joint_source_v2.zip'
PARITY = ROOT / ('BalloonPoppingGymEnv/evaluation/results/'
                 'pure_joint_standalone_parity_v2_20261009/summary.json')
FILES = (
    'review/scenario4_pure_joint_candidate_v2.py',
    'review/scenario4_pure_joint_candidate_v2.yaml',
    'review/scenario4_pure_joint_candidate_README.md',
    'BalloonPoppingGymEnv/agents/submission_time_allocation_v1.py',
    'BalloonPoppingGymEnv/agents/submission_scenario4_wind_profile_v1.py',
    'BalloonPoppingGymEnv/agents/submission_scenario4_guarded_innovation_v1.py',
    'BalloonPoppingGymEnv/agents/scenario4_wind_profile_agent.py',
    'BalloonPoppingGymEnv/agents/scenario4_forecast_innovation_agent.py',
    'BalloonPoppingGymEnv/agents/scenario4_guarded_innovation_agent.py',
    'BalloonPoppingGymEnv/agents/scenario4_pure_joint_refresh_agent.py',
    'BalloonPoppingGymEnv/agents/chain_beam_agent.py',
    'BalloonPoppingGymEnv/agents/chain_launch_agent.py',
    'BalloonPoppingGymEnv/evaluation/configs/scenario4_pure_joint_refresh_launch42.yaml',
    'scripts/build_scenario4_submission_agent.py',
    'scripts/build_scenario4_guarded_submission_agent.py',
    'scripts/build_scenario4_joint_submission_agent.py',
    'scripts/package_joint_candidate_review.py',
    'requirements.txt',
    'pyproject.toml',
    'uv.lock',
)


def main():
    parity = json.loads(PARITY.read_text())
    if not parity['passed'] or not all(parity['checks'].values()):
        raise RuntimeError('Standalone has not passed flight parity')
    if (ROOT / FILES[0]).read_text(encoding='utf-8') != build_source():
        raise RuntimeError('Source no longer matches the reproducible builder')
    flight = json.loads((PARITY.parent / 'flight.json').read_text())[0]
    if hashlib.sha256((ROOT / FILES[0]).read_bytes()).hexdigest() != flight['agent_sha256']:
        raise RuntimeError('Current standalone differs from the tested flight source')
    config = yaml.safe_load((ROOT / FILES[1]).read_text())
    tested = flight['config']
    if (config['scenario_number'] != tested['scenario_number'] or
            config['agent_class_name'] != tested['agent_class_name'] or
            config['agent_kwargs'] != tested['agent_kwargs'] or
            (ROOT / config['agent_module_path']).resolve() != (ROOT / FILES[0]).resolve()):
        raise RuntimeError('Review config differs from the tested policy')
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    contents = {}
    for name in FILES:
        path = ROOT / name
        if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(ROOT):
            raise ValueError(f'Invalid source input: {name}')
        raw = path.read_bytes()
        if re.search(rb'(?im)^\s*(?:team_secret|BPC_TEAM_SECRET)\s*[:=]\s*[\x22\x27]?[a-z0-9]{16,}', raw):
            raise ValueError(f'Credential-like assignment in {name}')
        contents[name] = raw
    manifest = dict(
        source_sha256={name: hashlib.sha256(raw).hexdigest() for name, raw in contents.items()},
        parity=parity,
        python=platform.python_version(),
        installed_packages={item.metadata['Name']: item.version
                            for item in metadata.distributions() if item.metadata['Name']},
        note='Local runtime inventory, not proof of official lockfile parity. No learned weights needed.')
    with ZipFile(OUTPUT, 'x', compression=ZIP_DEFLATED) as archive:
        for name, raw in sorted(contents.items()):
            archive.writestr(name, raw)
        archive.writestr('review/joint_source_manifest.json', json.dumps(manifest, indent=2))
    print(OUTPUT)
    print('sha256=' + hashlib.sha256(OUTPUT.read_bytes()).hexdigest())


if __name__ == '__main__':
    main()
