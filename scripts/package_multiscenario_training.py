"""Allowlisted runtime snapshot for the authorized training host, no credentials.

Includes exact local official simulator/data and pinned RocketPy source so the
remote job need not modify or reuse earlier projects. Excludes evaluation
results, submissions, credentials, old checkpoints, and Git metadata.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from zipfile import ZipFile, ZIP_DEFLATED
from scripts.train_multiscenario_residual_ppo import ROOT, source_hashes


def package(output):
    names = set(source_hashes())
    names.update([
        'BalloonPoppingGymEnv/__init__.py', 'BalloonPoppingGymEnv/console_logging.py',
        'BalloonPoppingGymEnv/envs/__init__.py',
        'BalloonPoppingGymEnv/evaluation/results/utils.py',
        'tests/test_multiscenario_residual.py', 'tests/test_official_residual_env.py',
        'scripts/package_multiscenario_training.py',
        'requirements.txt', 'pyproject.toml', 'LICENSE',
    ])
    for relative in ('scripts/__init__.py', 'tests/__init__.py',
                     'BalloonPoppingGymEnv/agents/__init__.py',
                     'BalloonPoppingGymEnv/training/__init__.py',
                     'BalloonPoppingGymEnv/evaluation/__init__.py',
                     'BalloonPoppingGymEnv/evaluation/results/__init__.py'):
        if (ROOT/relative).is_file():
            names.add(relative)
    names.update(str(p.relative_to(ROOT)).replace('\\', '/')
                 for p in (ROOT/'BalloonPoppingGymEnv/envs/data').glob('*.nc'))
    pinned = subprocess.check_output(['git', '-C', str(ROOT/'ActiveRocketPy'),
                                     'rev-parse', 'HEAD'], text=True).strip()
    runtime = subprocess.check_output(['git', '-C', str(ROOT/'ActiveRocketPy'),
                                      'ls-files', 'rocketpy'], text=True).splitlines()
    names.update('ActiveRocketPy/'+name for name in runtime)
    manifest = dict(rocketpy_commit=pinned, files={})
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(output, 'x', ZIP_DEFLATED) as archive:
        for name in sorted(names):
            path = ROOT/name
            if path.is_symlink() or not path.is_file() or ROOT not in path.resolve().parents:
                raise ValueError(f'Unsafe bundle path: {name}')
            content = path.read_bytes()
            manifest['files'][name] = hashlib.sha256(content).hexdigest()
            archive.writestr(name, content)
        archive.writestr('bundle_manifest.json', json.dumps(manifest, indent=2))
    return hashlib.sha256(output.read_bytes()).hexdigest(), len(names)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    digest, count = package(args.output)
    print(json.dumps(dict(path=str(args.output), sha256=digest, files=count)))
