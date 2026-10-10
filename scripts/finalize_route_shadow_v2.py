"""Finish an authorized long CPU collection after all pairs pass their gates.

This waits for the runner's summary or failure marker. It never starts flights
or uses a GPU. The output archive is a recoverable backup of generated data,
provenance, and the small source files needed to audit this collector.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
from zipfile import ZipFile, ZIP_DEFLATED

from scripts.summarize_route_shadow_v2 import summarize


ROOT = Path(__file__).resolve().parents[1]
SOURCES = (
    'BalloonPoppingGymEnv/training/route_shadow.py',
    'BalloonPoppingGymEnv/agents/route_attention_features.py',
    'scripts/collect_route_shadow.py',
    'scripts/audit_route_shadow.py',
    'scripts/summarize_route_shadow_v2.py',
    'scripts/finalize_route_shadow_v2.py',
    'tests/test_route_shadow.py',
    'tests/test_audit_route_shadow.py',
)


def finish(directory, archive, timeout_seconds=21600):
    directory, archive = Path(directory).resolve(), Path(archive).resolve()
    if not directory.is_dir() or archive.exists():
        raise ValueError('Expected an existing collection and a new archive path')
    start = time.monotonic()
    while not (directory/'summary.json').exists():
        if (directory/'failed.json').exists():
            raise RuntimeError(f'Collection failed; see {directory / "failed.json"}')
        if time.monotonic() - start >= timeout_seconds:
            raise TimeoutError('Collection did not complete within watcher budget')
        time.sleep(30)
    result = summarize(directory, require_complete=True)
    rollup = directory/'rollup.json'
    with rollup.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    paths = sorted(p for p in directory.rglob('*') if p.is_file())
    paths += [ROOT/name for name in SOURCES]
    with ZipFile(archive, 'x', ZIP_DEFLATED) as zipfile:
        for path in paths:
            if path.is_symlink() or not path.is_file():
                raise ValueError(f'Unsafe/incomplete backup file: {path}')
            name = (f'data/{path.relative_to(directory).as_posix()}'
                    if directory in path.parents else
                    f'source/{path.relative_to(ROOT).as_posix()}')
            zipfile.write(path, name)
    with ZipFile(archive) as zipfile:
        damaged = zipfile.testzip()
    if damaged is not None:
        raise RuntimeError(f'Backup integrity failure: {damaged}')
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    print(json.dumps(dict(complete=True, archive=str(archive),
                          archive_sha256=digest, fields=result['completed_fields'],
                          scores=result['scores'], mean_score=result['mean_score'],
                          total_incidental_pops=result['total_incidental_pops'])))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('archive', type=Path)
    parser.add_argument('--timeout-seconds', type=int, default=21600)
    args = parser.parse_args()
    finish(args.directory, args.archive, args.timeout_seconds)
