"""Conservative record of scenario seeds visible in local competition artifacts.

The registry includes executed seeds when logged and planned/literal seeds to
avoid accidental reuse. Older training jobs did not log every per-episode seed;
their exact fields cannot be certified by this file alone. No team credential
or raw submission content is copied into the registry.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import re
import tarfile
import tempfile
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT/'BalloonPoppingGymEnv/evaluation/results'
QUALIFICATION = ROOT/'qualification'
SEED_KEYS = {'seed', 'random_seed', 'np_random_seed', 'episode_seed',
             'anchor_seed', 'seeds', 'development_seeds', 'validation_seeds',
             'holdout_seeds', 'initial_seed', 'eval_seed'}
SEED_TEXT = re.compile(r'(?i)\b(?:random_?seed|episode_?seed|seed)[_ :=-]*(\d{1,20})\b')
MAX_STRUCTURED_BYTES = 8_000_000


def _is_seed(value):
    return type(value) is int and 0 <= value < 2**64


class SeedRegistry:
    def __init__(self):
        self.origins: dict[int, set[str]] = {}
        self.warnings: set[str] = set()
        self.files_checked = 0

    def add(self, value, origin):
        if _is_seed(value):
            self.origins.setdefault(value, set()).add(origin)

    def structured(self, value, origin):
        if isinstance(value, dict):
            for key, child in value.items():
                key_text = str(key).lower()
                if key_text in SEED_KEYS:
                    if _is_seed(child):
                        self.add(child, origin)
                    elif isinstance(child, (tuple, list)):
                        for item in child:
                            self.add(item, origin)
                self.structured(child, origin)
        elif isinstance(value, (tuple, list)):
            for child in value:
                self.structured(child, origin)

    def text(self, value, origin):
        for match in SEED_TEXT.finditer(value):
            self.add(int(match.group(1)), origin)

    def json_bytes(self, raw, origin):
        if len(raw) > MAX_STRUCTURED_BYTES:
            self.warnings.add(f'Skipped oversized structured artifact: {origin}')
            return
        try:
            self.structured(json.loads(raw), origin)
        except (UnicodeError, ValueError):
            self.warnings.add(f'Could not parse seed artifact: {origin}')

    def jsonl_bytes(self, raw, origin):
        if len(raw) > MAX_STRUCTURED_BYTES:
            self.warnings.add(f'Skipped oversized seed log: {origin}')
            return
        found = False
        for line in raw.splitlines():
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except (UnicodeError, ValueError):
                self.warnings.add(f'Could not parse a log line: {origin}')
                continue
            found |= isinstance(value, dict) and any(
                str(key).lower() in SEED_KEYS for key in value)
            self.structured(value, origin)
        if raw.strip() and not found and Path(origin).name == 'episodes.jsonl':
            self.warnings.add(f'Legacy episode log lacks field seeds: {origin}')

    def submission_header(self, raw, origin):
        # The packed JSON's leaderboard_info object precedes the enormous
        # trajectory. Parse only that small object, never the team/secret body.
        head = raw[:64_000].decode('utf-8-sig', errors='replace')
        match = re.search(r'"leaderboard_info"\s*:\s*', head)
        if match is None:
            self.warnings.add(f'No leaderboard header in submission: {origin}')
            return
        try:
            info, _ = json.JSONDecoder().raw_decode(head, match.end())
            self.structured(info, origin)
        except ValueError:
            self.warnings.add(f'Could not parse leaderboard header: {origin}')

    def artifact(self, name, raw):
        lower = name.lower().replace('\\', '/')
        base = lower.rsplit('/', 1)[-1]
        self.text(base, name)
        if 'trajectory' in base or base.endswith(('.py', '.npz', '.pkl')):
            return
        if base.endswith('_submission.json'):
            self.submission_header(raw, name)
        elif base.endswith('.jsonl') and any(
                token in base for token in ('seed', 'episode', 'reset')):
            self.jsonl_bytes(raw, name)
        elif base.endswith('.json') and any(token in base for token in (
                'seed', 'plan', 'progress', 'summary', 'manifest', 'metric',
                'report', 'result', 'eval', 'receipt', 'started')):
            self.json_bytes(raw, name)
        elif base.endswith('.log') and len(raw) <= MAX_STRUCTURED_BYTES:
            self.text(raw.decode('utf-8', errors='replace'), name)

    def source_literals(self, path):
        origin = str(path.relative_to(ROOT)).replace('\\', '/')
        try:
            source = path.read_text(encoding='utf-8-sig')
        except (OSError, UnicodeError):
            self.warnings.add(f'Could not read source: {origin}')
            return
        self.text(source, origin)
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                names = [t.id for t in node.targets if isinstance(t, ast.Name)]
                value = node.value
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                names, value = [node.target.id], node.value
            else:
                continue
            if not any('SEED' in name.upper() for name in names):
                continue
            try:
                literal = ast.literal_eval(value)
            except (ValueError, TypeError, SyntaxError, MemoryError):
                continue
            if _is_seed(literal):
                self.add(literal, origin)
            elif isinstance(literal, (tuple, list)):
                for item in literal:
                    self.add(item, origin)

    def as_dict(self):
        return {
            'schema': 'qualification_used_seed_registry_v1',
            'note': 'Conservative local evidence, not proof of every legacy training episode',
            'seed_count': len(self.origins),
            'files_checked': self.files_checked,
            'seeds': {str(seed): sorted(origins) for seed, origins in sorted(self.origins.items())},
            'warnings': sorted(self.warnings),
        }


def collect(root=ROOT):
    root = Path(root).resolve()
    registry = SeedRegistry()
    results = root/'BalloonPoppingGymEnv/evaluation/results'
    if not results.is_dir():
        raise FileNotFoundError(results)
    for path in sorted(results.iterdir()):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        label = str(path.relative_to(root)).replace('\\', '/')
        registry.files_checked += 1
        registry.text(path.name, label)
        if suffix in {'.json', '.jsonl', '.log'}:
            if 'trajectory' in path.name.lower():
                continue
            if suffix == '.json' and path.stat().st_size > MAX_STRUCTURED_BYTES:
                if path.name.endswith('_submission.json'):
                    with path.open('rb') as stream:
                        registry.submission_header(stream.read(64_000), label)
                else:
                    registry.warnings.add(f'Skipped oversized local JSON: {label}')
                continue
            registry.artifact(label, path.read_bytes())
        elif suffix == '.zip':
            try:
                with zipfile.ZipFile(path) as archive:
                    for entry in archive.infolist():
                        if entry.is_dir() or entry.file_size > MAX_STRUCTURED_BYTES:
                            continue
                        item = f'{label}!{entry.filename}'
                        if not _interesting(entry.filename):
                            registry.text(entry.filename, item)
                            continue
                        registry.files_checked += 1
                        registry.artifact(item, archive.read(entry))
            except (OSError, zipfile.BadZipFile):
                registry.warnings.add(f'Could not inspect archive: {label}')
        elif suffix == '.tgz':
            try:
                legacy_runs = {}
                with tarfile.open(path, 'r:gz') as archive:
                    for entry in archive:
                        if not entry.isfile() or entry.size > MAX_STRUCTURED_BYTES:
                            continue
                        item = f'{label}!{entry.name}'
                        folder, _, base = entry.name.replace('\\', '/').rpartition('/')
                        monitor = re.fullmatch(r'worker_(\d+)\.monitor\.csv', base)
                        if not _interesting(entry.name) and monitor is None:
                            registry.text(entry.name, item)
                            continue
                        stream = archive.extractfile(entry)
                        if stream is not None:
                            raw = stream.read()
                            metadata = legacy_runs.setdefault(folder, dict(monitors={}, failures={}))
                            if base == 'manifest.json':
                                try:
                                    metadata['settings'] = json.loads(raw).get('settings', {})
                                except ValueError:
                                    pass
                            if monitor:
                                rows = [line for line in raw.decode('utf-8', errors='replace').splitlines()
                                        if line and not line.startswith('#')]
                                metadata['monitors'][int(monitor.group(1))] = max(0, len(rows)-1)
                            failure = re.fullmatch(r'reset_failures_(\d+)\.jsonl', base)
                            if failure:
                                metadata['failures'][int(failure.group(1))] = sum(bool(line.strip())
                                    for line in raw.splitlines())
                            if base == 'episodes.jsonl' and b'"seed"' not in raw:
                                metadata['unlogged_episodes'] = True
                            registry.files_checked += 1
                            registry.artifact(item, raw)
                for folder, metadata in legacy_runs.items():
                    if not metadata.get('unlogged_episodes'):
                        continue
                    settings = metadata.get('settings', {})
                    base_seed, workers = settings.get('seed'), settings.get('envs')
                    if not (_is_seed(base_seed) and type(workers) is int and 1 <= workers <= 32):
                        registry.warnings.add(f'Cannot reconstruct legacy seed stream: {label}!{folder}')
                        continue
                    for rank in range(workers):
                        # A monitor row is one completed episode. Add the active
                        # unfinished episode, every recorded failed reset and ten
                        # extra draws; over-reserving is safer than missing a seed.
                        draws = (metadata['monitors'].get(rank, 0)
                                 + metadata['failures'].get(rank, 0) + 11)
                        rng = np.random.default_rng(base_seed+rank)
                        origin = f'{label}!{folder}/legacy_rng_worker_{rank}_first_{draws}'
                        for value in rng.integers(100_000, 2**31-1, size=draws):
                            registry.add(int(value), origin)
                    registry.warnings.add(
                        f'Conservative RNG prefix reconstructed, exact episodes unverified: {label}!{folder}')
            except (OSError, tarfile.TarError):
                registry.warnings.add(f'Could not inspect archive: {label}')
    for folder in ('scripts', 'tests', 'BalloonPoppingGymEnv/agents'):
        for path in sorted((root/folder).glob('*.py')):
            registry.source_literals(path)
    for path in sorted((root/'BalloonPoppingGymEnv/evaluation/configs').glob('*.yaml')):
        origin = str(path.relative_to(root)).replace('\\', '/')
        registry.text(path.read_text(encoding='utf-8-sig'), origin)
    scenario = root/'BalloonPoppingGymEnv/envs/scenario_parameters/scenario_4_parameters.yaml'
    registry.text(scenario.read_text(encoding='utf-8-sig'), str(scenario.relative_to(root)))
    qualification = root/'qualification'
    if qualification.is_dir():
        for path in sorted(qualification.rglob('*.json')):
            if path.name == 'used_seeds.json':
                continue
            if path.stat().st_size <= MAX_STRUCTURED_BYTES:
                label = str(path.relative_to(root)).replace('\\', '/')
                registry.json_bytes(path.read_bytes(), label)
        manual = qualification/'used_seed_manual.jsonl'
        if manual.exists():
            registry.jsonl_bytes(manual.read_bytes(), str(manual.relative_to(root)))
    return registry


def _interesting(name):
    base = name.lower().replace('\\', '/').rsplit('/', 1)[-1]
    if 'trajectory' in base or base.endswith(('.py', '.npz', '.pkl', '.zip')):
        return False
    return base.endswith(('.json', '.jsonl', '.log')) and any(token in base for token in (
        'seed', 'plan', 'progress', 'summary', 'manifest', 'metric', 'report',
        'result', 'eval', 'receipt', 'started', 'episode', 'reset'))


def save(registry, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(registry.as_dict(), ensure_ascii=False, indent=2)+'\n'
    handle, temporary = tempfile.mkstemp(dir=path.parent, prefix='.used_seeds_', suffix='.tmp')
    try:
        with os.fdopen(handle, 'w', encoding='utf-8') as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return hashlib.sha256(content.encode('utf-8')).hexdigest()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=QUALIFICATION/'used_seeds.json')
    parser.add_argument('--check', type=int)
    args = parser.parse_args()
    found = collect()
    digest = save(found, args.output)
    print(f'Recorded {len(found.origins)} distinct local/planned seeds; registry SHA256 {digest}')
    print(f'Coverage warnings: {len(found.warnings)} (see the registry file)')
    if args.check is not None:
        matches = found.origins.get(args.check, set())
        print('USED_OR_RESERVED' if matches else 'NOT_FOUND_IN_LOCAL_RECORD')
        for origin in sorted(matches)[:5]:
            print(origin)
