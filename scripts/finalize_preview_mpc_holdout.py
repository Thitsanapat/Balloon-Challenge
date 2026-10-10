"""Audit and archive the ten-seed MPC+PPO holdout without promoting it."""
import argparse
import hashlib
import json
from pathlib import Path
import time
from zipfile import ZipFile, ZIP_DEFLATED

from scripts.finalize_preview_mpc_validation import summarize

ROOT = Path(__file__).resolve().parents[1]
SOURCE_NAMES = (
    'BalloonPoppingGymEnv/agents/preview_mpc.py',
    'BalloonPoppingGymEnv/agents/preview_mpc_ppo_agent.py',
    'scripts/compare_preview_mpc.py',
    'scripts/validate_preview_mpc.py',
    'scripts/holdout_preview_mpc.py',
    'scripts/finalize_preview_mpc_validation.py',
    'scripts/finalize_preview_mpc_holdout.py',
    'tests/test_preview_mpc.py',
)


def finish(directory, archive, timeout_seconds=28800):
    directory, archive = Path(directory).resolve(), Path(archive).resolve()
    if not directory.is_dir() or archive.exists():
        raise ValueError('Existing collection and new archive path required')
    start = time.monotonic()
    while not (directory/'summary.json').exists():
        if (directory/'failed.json').exists():
            raise RuntimeError(f'Holdout failed; inspect {directory / "failed.json"}')
        if time.monotonic()-start >= timeout_seconds:
            raise TimeoutError('Holdout did not finish in watcher budget')
        time.sleep(30)
    report = summarize(directory)
    report['interpretation'] = 'Ten unseen paired seeds; no live-score guarantee'
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
