"""Summarize and back up a completed, matched official MPC pilot."""
import argparse
import hashlib
import json
from pathlib import Path
import time
from zipfile import ZipFile, ZIP_DEFLATED

from scripts.collect_route_shadow import paired_gate

ROOT = Path(__file__).resolve().parents[1]
SOURCE_NAMES = (
    'BalloonPoppingGymEnv/agents/preview_mpc.py',
    'BalloonPoppingGymEnv/agents/preview_mpc_ppo_agent.py',
    'scripts/compare_preview_mpc.py',
    'scripts/finalize_preview_mpc.py',
    'tests/test_preview_mpc.py',
)


def summarize(directory):
    directory = Path(directory)
    plan = json.loads((directory/'plan.json').read_text(encoding='utf-8'))
    summary = json.loads((directory/'summary.json').read_text(encoding='utf-8'))
    rows = summary['completed']
    expected = [plan['anchor_seed']] + plan['development_seeds']
    if not summary['all_complete'] or [r['seed'] for r in rows] != expected:
        raise ValueError('Incomplete or reordered MPC comparison')
    if not paired_gate(rows[0]['parent'], rows[0]['bypass']):
        raise ValueError('Zero-gain MPC failed action/observation parity')
    for path, sha in plan['frozen_sha256'].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != sha:
            raise ValueError(f'Frozen source/model changed: {path}')
    for row in rows[1:]:
        for label in ('parent', 'mpc'):
            trial = row[label]
            if not trial['terminated'] or trial['truncated']:
                raise ValueError('Incomplete official flight')
        if row['mpc']['mpc_diagnostics']['solves'] < 1:
            raise ValueError('MPC did not actually solve on a development field')
    baseline = [r['parent']['score'] for r in rows[1:]]
    candidate = [r['mpc']['score'] for r in rows[1:]]
    differences = [a-b for a, b in zip(candidate, baseline)]
    return dict(complete=True, anchor_parity=True,
                development_seeds=plan['development_seeds'], parent_scores=baseline,
                mpc_ppo_scores=candidate, differences=differences,
                parent_mean=sum(baseline)/len(baseline),
                mpc_ppo_mean=sum(candidate)/len(candidate),
                wins=sum(d > 0 for d in differences),
                ties=sum(d == 0 for d in differences),
                losses=sum(d < 0 for d in differences),
                mpc_solves=[r['mpc']['mpc_diagnostics']['solves'] for r in rows[1:]],
                interpretation='Small predeclared development pilot; no held-out promotion')


def finish(directory, archive, timeout_seconds=14400):
    directory, archive = Path(directory).resolve(), Path(archive).resolve()
    if not directory.is_dir() or archive.exists():
        raise ValueError('Existing collection and new archive path required')
    start = time.monotonic()
    while not (directory/'summary.json').exists():
        if (directory/'failed.json').exists():
            raise RuntimeError(f'MPC pilot failed; inspect {directory / "failed.json"}')
        if time.monotonic()-start >= timeout_seconds:
            raise TimeoutError('MPC comparison did not finish in watcher budget')
        time.sleep(30)
    report = summarize(directory)
    with (directory/'rollup.json').open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    paths = sorted(p for p in directory.rglob('*') if p.is_file())
    paths += [ROOT/name for name in SOURCE_NAMES]
    with ZipFile(archive, 'x', ZIP_DEFLATED) as zipfile:
        for path in paths:
            if path.is_symlink() or not path.is_file():
                raise ValueError(f'Unsafe archive entry: {path}')
            name = (f'data/{path.relative_to(directory).as_posix()}'
                    if directory in path.parents else
                    f'source/{path.relative_to(ROOT).as_posix()}')
            zipfile.write(path, name)
    with ZipFile(archive) as zipfile:
        damaged = zipfile.testzip()
    if damaged is not None:
        raise RuntimeError(f'Archive integrity failure: {damaged}')
    print(json.dumps(dict(report=report, archive=str(archive),
                          archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest())))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('archive', type=Path)
    args = parser.parse_args()
    finish(args.directory, args.archive)
