"""Archive the frozen, locally rehearsed joint agent; exclude credentials/results JSON."""
import ast
import hashlib
from importlib import metadata
import json
from pathlib import Path
import platform
import re
from zipfile import ZIP_DEFLATED, ZipFile

import yaml
from scripts.build_scenario4_joint_submission_agent import build_source
from scripts.package_joint_candidate_review import FILES
from scripts.validate_joint_live_ready import CONFIG, EXPECTED, REPORT

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'review/scenario4_finalnight_joint_source_20261010.zip'
EXTRA = (
    'BalloonPoppingGymEnv/agents/submission_scenario4_pure_joint_v2.py',
    'BalloonPoppingGymEnv/evaluation/configs/submission_scenario4_pure_joint_launch42.yaml',
    'BalloonPoppingGymEnv/evaluation/configs/submission_scenario4_guarded_innovation_launch42.yaml',
    'scripts/qualification_run.py', 'scripts/qualification_seed_registry.py',
    'scripts/run_submission.py', 'scripts/validate_joint_live_ready.py',
    'scripts/evaluate_guidance.py', 'scripts/guidance_telemetry.py',
    'scripts/package_finalnight_joint.py',
    'run_qualification.cmd', 'run_qualification_guarded_backup.cmd',
    'doc/scenario4_finalnight_20261010.md', 'doc/qualification_runbook_20261007.md',
    'tests/test_scenario4_joint_submission_build.py',
    'tests/test_scenario4_pure_joint_refresh.py',
)


def main():
    readiness = json.loads(REPORT.read_text(encoding='utf-8'))
    source = ROOT / EXTRA[0]
    if (not readiness['passed'] or not readiness['scenario_restored'] or
            readiness['result']['agent_sha256'] != EXPECTED or
            hashlib.sha256(source.read_bytes()).hexdigest() != EXPECTED or
            source.read_text(encoding='utf-8') != build_source()):
        raise RuntimeError('Standalone/readiness verification failed')
    from scripts.qualification_run import DEFAULT_CONFIG
    if DEFAULT_CONFIG != CONFIG:
        raise RuntimeError('Default launcher has not selected the verified joint config')
    reviewed = yaml.safe_load((ROOT / 'review/scenario4_pure_joint_candidate_v2.yaml').read_text())
    selected = yaml.safe_load(CONFIG.read_text())
    for key in ('agent_kwargs', 'scenario_number', 'agent_class_name'):
        if selected[key] != reviewed[key]:
            raise RuntimeError(f'Selected config changed: {key}')
    names = set(FILES + EXTRA)
    # Include modular agent dependencies too, so the bundled unit tests and
    # reference implementation work outside this development checkout.
    pending = list(names)
    while pending:
        name = pending.pop()
        if not name.endswith('.py'):
            continue
        for node in ast.walk(ast.parse((ROOT / name).read_text(encoding='utf-8-sig'))):
            if (isinstance(node, ast.ImportFrom) and node.module and
                    node.module.startswith('BalloonPoppingGymEnv.agents.')):
                dependency = node.module.replace('.', '/') + '.py'
                if dependency not in names:
                    names.add(dependency)
                    pending.append(dependency)
    contents = {name: (ROOT / name).read_bytes() for name in sorted(names)}
    for name, raw in contents.items():
        if re.search(rb'(?im)^\s*(?:team_secret|BPC_TEAM_SECRET)\s*[:=]\s*[\x22\x27]?[a-z0-9]{16,}', raw):
            raise ValueError(f'Credential-like assignment in {name}')
    manifest = dict(
        files={name: hashlib.sha256(raw).hexdigest() for name, raw in contents.items()},
        agent_sha256=EXPECTED, python=platform.python_version(),
        installed_packages={d.metadata['Name']: d.version for d in metadata.distributions()
                            if d.metadata['Name']},
        local_packaging_test=dict(seed=readiness['result']['seed'],
                                  score=readiness['result']['score'],
                                  submission_sha256=readiness['result']['submission_sha256'],
                                  uploaded=False),
        note='Source only; install over official v0.2.2 with its submodule. Runtime inventory is not lockfile parity.')
    with ZipFile(OUTPUT, 'x', compression=ZIP_DEFLATED) as archive:
        for name, raw in contents.items():
            archive.writestr(name, raw)
        archive.writestr('review/finalnight_manifest.json', json.dumps(manifest, indent=2))
    with ZipFile(OUTPUT) as archive:
        if archive.testzip() is not None:
            raise RuntimeError('Archive CRC failed')
        for name, digest in manifest['files'].items():
            if hashlib.sha256(archive.read(name)).hexdigest() != digest:
                raise RuntimeError(f'Archive hash mismatch: {name}')
    print(OUTPUT)
    print('SHA256=' + hashlib.sha256(OUTPUT.read_bytes()).hexdigest())
    print(f'{len(contents)} files verified; no submission JSON or credentials included.')


if __name__ == '__main__':
    main()
