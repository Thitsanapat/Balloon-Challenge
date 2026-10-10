"""Audit and archive the predeclared unseen MPC+PPO validation pairs."""
import argparse
import hashlib
import json
from pathlib import Path
import time
from zipfile import ZipFile, ZIP_DEFLATED

ROOT = Path(__file__).resolve().parents[1]
SOURCE_NAMES = (
    'BalloonPoppingGymEnv/agents/preview_mpc.py',
    'BalloonPoppingGymEnv/agents/preview_mpc_ppo_agent.py',
    'scripts/compare_preview_mpc.py',
    'scripts/validate_preview_mpc.py',
    'scripts/finalize_preview_mpc_validation.py',
    'tests/test_preview_mpc.py',
)


def summarize(directory):
    directory = Path(directory)
    plan = json.loads((directory/'plan.json').read_text(encoding='utf-8'))
    summary = json.loads((directory/'summary.json').read_text(encoding='utf-8'))
    rows = summary['completed']
    if not summary['all_complete'] or [r['seed'] for r in rows] != plan['seeds']:
        raise ValueError('Incomplete or reordered validation pairs')
    for path, sha in plan['frozen_sha256'].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != sha:
            raise ValueError(f'Frozen source/model changed: {path}')
    for row in rows:
        for label in ('parent', 'mpc'):
            trial = row[label]
            if (trial['seed'] != row['seed'] or not trial['terminated'] or trial['truncated']
                    or len(trial['pop_events']) != trial['score']):
                raise ValueError(f'Invalid official flight: {row["seed"]} {label}')
        if row['mpc']['mpc_diagnostics']['solves'] < 1:
            raise ValueError('MPC did not solve on a validation field')
    baseline = [r['parent']['score'] for r in rows]
    candidate = [r['mpc']['score'] for r in rows]
    differences = [a-b for a, b in zip(candidate, baseline)]
    return dict(complete=True, validation_seeds=plan['seeds'],
                parent_scores=baseline, mpc_ppo_scores=candidate,
                differences=differences, parent_mean=sum(baseline)/len(baseline),
                mpc_ppo_mean=sum(candidate)/len(candidate),
                wins=sum(d > 0 for d in differences),
                ties=sum(d == 0 for d in differences),
                losses=sum(d < 0 for d in differences),
                interpretation='Five unseen paired seeds; not a tournament-score guarantee')


def finish(directory, archive, timeout_seconds=28800):
    directory, archive = Path(directory).resolve(), Path(archive).resolve()
    if not directory.is_dir() or archive.exists():
        raise ValueError('Existing collection and new archive path required')
    start = time.monotonic()
    while not (directory/'summary.json').exists():
        if (directory/'failed.json').exists():
            raise RuntimeError(f'Validation failed; inspect {directory / "failed.json"}')
        if time.monotonic()-start >= timeout_seconds:
            raise TimeoutError('Validation did not finish in watcher budget')
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
