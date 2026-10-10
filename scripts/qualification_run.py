"""Interactive, reversible Scenario-4 qualification run using official evaluator.

The only temporary scenario edit is scenario.random_seed. The original YAML is
restored byte-for-byte even when evaluation fails normally. The packed JSON is
created by the repository's existing official packer and is never uploaded by
this script. Team credentials are never printed or stored in receipts.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from uuid import uuid4

import yaml

from scripts.qualification_seed_registry import ROOT, RESULTS, QUALIFICATION, collect, save

DEFAULT_CONFIG = ROOT/'BalloonPoppingGymEnv/evaluation/configs/submission_scenario4_pure_joint_launch42.yaml'
SCENARIO_FILE = ROOT/'BalloonPoppingGymEnv/envs/scenario_parameters/scenario_4_parameters.yaml'
RUNTIME = QUALIFICATION/'.runtime'
RECEIPTS = QUALIFICATION/'receipts'
ACTIVE = RUNTIME/'active_seed_edit.json'
SEED_LINE = re.compile(rb'(?m)^([ \t]*random_seed:[ \t]*)(?:null|[0-9]+)([ \t]*(?:\#[^\r\n]*)?)(?=\r?$)')
OFFICIAL_FILES = (
    'BalloonPoppingGymEnv/envs/scenario_parameters/scenario_4_parameters.yaml',
    'BalloonPoppingGymEnv/envs/scenario_parameters/scenario_4_given_parameters.yaml',
    'BalloonPoppingGymEnv/envs/data/TW_Cup_260801_NetCDF4_Ensemble.nc',
    'BalloonPoppingGymEnv/evaluation/evaluate.py',
    'BalloonPoppingGymEnv/evaluation/results/utils.py',
    'BalloonPoppingGymEnv/envs/balloon_world.py',
)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def atomic_bytes(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=path.parent, prefix='.qualification_', suffix='.tmp')
    try:
        with os.fdopen(handle, 'wb') as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def replace_seed(original, seed):
    if not (type(seed) is int and 0 <= seed < 2**64):
        raise ValueError('Seed must be one nonnegative integer below 2^64')
    matches = list(SEED_LINE.finditer(original))
    if len(matches) != 1:
        raise ValueError('Expected exactly one scenario random_seed line')
    match = matches[0]
    changed = original[:match.start()] + match.group(1) + str(seed).encode('ascii') \
        + match.group(2) + original[match.end():]
    before = yaml.safe_load(original.decode('utf-8-sig'))
    after = yaml.safe_load(changed.decode('utf-8-sig'))
    before['scenario']['random_seed'] = seed
    if before != after or after['scenario']['number'] != 4:
        raise ValueError('Seed replacement changed another scenario parameter')
    return changed


def verify_official_files():
    """Compare the evaluator and installed flight physics to the v0.2.2 tag."""
    for name in OFFICIAL_FILES:
        tagged = subprocess.run(['git', 'show', f'v0.2.2:{name}'], cwd=ROOT,
                                capture_output=True, check=True).stdout
        local = (ROOT/name).read_bytes()
        if name.endswith('scenario_4_parameters.yaml'):
            tagged, local = replace_seed(tagged, 0), replace_seed(local, 0)
        normalize = lambda raw: raw.removeprefix(b'\xef\xbb\xbf').replace(b'\r\n', b'\n')
        if normalize(tagged) != normalize(local):
            raise RuntimeError(f'Official v0.2.2 file differs beyond the seed: {name}')
    expected = subprocess.run(['git', 'rev-parse', 'v0.2.2:ActiveRocketPy'],
                              cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    actual = subprocess.run(['git', '-C', str(ROOT/'ActiveRocketPy'), 'rev-parse', 'HEAD'],
                            cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    if expected != actual:
        raise RuntimeError('ActiveRocketPy is not at the v0.2.2 recorded commit')
    spec = importlib.util.find_spec('rocketpy')
    if spec is None or spec.origin is None:
        raise RuntimeError('ActiveRocketPy/rocketpy is not installed in this Python environment')
    installed = Path(spec.origin).resolve().parent
    pinned = ROOT/'ActiveRocketPy/rocketpy'
    pinned_files = {path.relative_to(pinned) for path in pinned.rglob('*.py')}
    installed_files = {path.relative_to(installed) for path in installed.rglob('*.py')}
    if not pinned_files or pinned_files != installed_files:
        raise RuntimeError('Installed rocketpy Python file set differs from the pinned ActiveRocketPy')
    for relative in sorted(pinned_files):
        if (pinned/relative).read_bytes() != (installed/relative).read_bytes():
            raise RuntimeError(f'Installed rocketpy differs from pinned ActiveRocketPy: {relative}')


def check_config(path):
    path = Path(path).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError('Use a configuration inside this workspace')
    config = yaml.safe_load(path.read_text(encoding='utf-8-sig'))
    if (not isinstance(config, dict) or config.get('scenario_number') != 4
            or config.get('render_mode') is not None
            or not isinstance(config.get('agent_kwargs'), dict)):
        raise ValueError('Live configuration must be Scenario 4, headless, with agent_kwargs')
    source = (ROOT/config['agent_module_path']).resolve()
    agents = ROOT/'BalloonPoppingGymEnv/agents'
    if not source.is_relative_to(agents) or source.suffix != '.py' or not source.is_file():
        raise ValueError('Configured agent source must be an existing /agents Python file')
    return config, source


def parse_small_object_prefix(path, key):
    with Path(path).open('rb') as stream:
        head = stream.read(64_000).decode('utf-8-sig', errors='replace')
    match = re.search(r'"'+re.escape(key)+r'"\s*:\s*', head)
    if match is None:
        raise ValueError(f'Missing {key} object in JSON header')
    value, _ = json.JSONDecoder().raw_decode(head, match.end())
    if not isinstance(value, dict):
        raise ValueError(f'{key} must be a JSON object')
    return value


def credentials_from_previous_submission():
    candidates = sorted(RESULTS.glob('*_submission.json'),
                        key=lambda path: path.stat().st_mtime, reverse=True)
    for path in candidates:
        try:
            team = parse_small_object_prefix(path, 'team')
            if (isinstance(team.get('name'), str) and team['name']
                    and isinstance(team.get('secret'), str) and team['secret']):
                return team['name'], team['secret'], path.name
        except (OSError, ValueError):
            continue
    raise RuntimeError('No earlier local submission with credentials; provide --credentials-file')


def prepare_credentials(path, environment):
    if path is not None:
        path = Path(path).resolve()
        if path.suffix.lower() == '.json':
            team = parse_small_object_prefix(path, 'team')
            name, secret = team.get('name'), team.get('secret')
        else:
            value = path.read_text(encoding='utf-8-sig').replace('\\_', '_')
            found = {key: re.search(rf'(?m)^\s*{key}\s*:\s*([^\s]+)\s*$', value)
                     for key in ('team_name', 'team_secret')}
            name = found['team_name'].group(1) if found['team_name'] else None
            secret = found['team_secret'].group(1) if found['team_secret'] else None
        source = path.name
    elif environment.get('BPC_TEAM_NAME') and environment.get('BPC_TEAM_SECRET'):
        name, secret, source = environment['BPC_TEAM_NAME'], environment['BPC_TEAM_SECRET'], 'environment'
    else:
        name, secret, source = credentials_from_previous_submission()
    if not (isinstance(name, str) and name and isinstance(secret, str) and secret):
        raise ValueError('Team credentials are incomplete')
    environment['BPC_TEAM_NAME'] = name
    environment['BPC_TEAM_SECRET'] = secret
    return name, source


@contextmanager
def temporary_seed(seed, run_id):
    if ACTIVE.exists():
        raise RuntimeError(f'Previous seed edit needs recovery: {ACTIVE}; use --recover')
    RUNTIME.mkdir(parents=True, exist_ok=True)
    original = SCENARIO_FILE.read_bytes()
    changed = replace_seed(original, seed)
    backup = RUNTIME/f'{run_id}_scenario4_original.yaml'
    with backup.open('xb') as stream:
        stream.write(original)
        stream.flush()
        os.fsync(stream.fileno())
    journal = dict(seed=seed, path=str(SCENARIO_FILE), backup=str(backup),
                   original_sha256=hashlib.sha256(original).hexdigest(),
                   changed_sha256=hashlib.sha256(changed).hexdigest(), pid=os.getpid())
    atomic_bytes(ACTIVE, json.dumps(journal, indent=2).encode('utf-8'))
    try:
        atomic_bytes(SCENARIO_FILE, changed)
        yield
    finally:
        current = SCENARIO_FILE.read_bytes()
        if current not in (original, changed):
            raise RuntimeError(f'Scenario file changed during run; preserved backup at {backup}')
        atomic_bytes(SCENARIO_FILE, original)
        if SCENARIO_FILE.read_bytes() != original:
            raise RuntimeError(f'Scenario restoration failed; preserved backup at {backup}')
        ACTIVE.unlink()


def recover_seed_edit():
    if not ACTIVE.exists():
        print('No interrupted seed edit found.')
        return
    journal = json.loads(ACTIVE.read_text(encoding='utf-8'))
    if Path(journal['path']).resolve() != SCENARIO_FILE.resolve():
        raise RuntimeError('Recovery journal targets another file; inspect manually')
    backup = Path(journal['backup']).resolve()
    if not backup.is_relative_to(RUNTIME) or sha256_file(backup) != journal['original_sha256']:
        raise RuntimeError('Recovery backup mismatch; inspect manually')
    current = sha256_file(SCENARIO_FILE)
    if current not in (journal['original_sha256'], journal['changed_sha256']):
        raise RuntimeError('Scenario changed unexpectedly; no automatic overwrite')
    atomic_bytes(SCENARIO_FILE, backup.read_bytes())
    ACTIVE.unlink()
    print('Original Scenario 4 YAML restored byte-for-byte. Backup retained for audit.')


def reserve_seed(seed, run_id, mode, agent_sha):
    QUALIFICATION.mkdir(parents=True, exist_ok=True)
    record = dict(seed=seed, usage=f'qualification_{mode}_attempt', run_id=run_id,
                  agent_sha256=agent_sha,
                  started_utc=datetime.now(timezone.utc).isoformat())
    path = QUALIFICATION/'used_seed_manual.jsonl'
    with path.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(record)+'\n')
        stream.flush()
        os.fsync(stream.fileno())


def validate_submission(path, seed, score, team_name):
    # A full parse catches malformed/truncated JSON, including the former
    # "submission must be a JSON object" failure. Never print payload fields.
    with Path(path).open('r', encoding='utf-8') as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise ValueError('Submission is not a JSON object')
    info = payload.get('leaderboard_info', {})
    world = payload.get('balloon_world_data', {})
    if (payload.get('format_version') != 2 or
            info.get('scenario_number') != 4 or
            info.get('random_seed') != seed or
            info.get('final_reward') != score or
            info.get('episode_ending') is None or
            world.get('scenario_parameters', {}).get('scenario', {}).get('random_seed') != seed or
            payload.get('team', {}).get('name') != team_name or
            not payload.get('team', {}).get('secret') or
            not isinstance(payload.get('agent_info', {}).get('agent_module_file'), str)):
        raise ValueError('Saved submission metadata did not match this evaluation')
    return sha256_file(path)


def run(seed, config_path, mode, credentials_file=None):
    if ACTIVE.exists():
        raise RuntimeError(f'Interrupted or active run detected: {ACTIVE}; do not run twice')
    verify_official_files()
    config, agent_source = check_config(config_path)
    registry = collect(ROOT)
    if seed in registry.origins:
        locations = sorted(registry.origins[seed])[:5]
        raise ValueError(f'Seed {seed} has been used/reserved locally. Ask organizer to redraw. '
                         f'Evidence: {locations}')
    registry_sha = save(registry, QUALIFICATION/'used_seeds.json')
    free_bytes = shutil.disk_usage(ROOT).free
    if free_bytes < 1_000_000_000:
        raise RuntimeError('Less than 1 GB free disk space; evaluation may leave an incomplete result')
    env = dict(os.environ)
    team_name = credential_source = None
    if mode == 'live':
        team_name, credential_source = prepare_credentials(credentials_file, env)
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+f'_seed{seed}_{uuid4().hex[:8]}'
    RECEIPTS.mkdir(parents=True, exist_ok=True)
    official_report = RECEIPTS/f'{run_id}_official_receipt.json'
    summary_path = RECEIPTS/f'{run_id}_summary.json'
    source_sha = sha256_file(agent_source)
    print(f'Scenario 4 v0.2.2 verified | mode={mode} | seed={seed}')
    print(f'Agent: {agent_source.name} | SHA256={source_sha}')
    print(f'Seed registry: {len(registry.origins)} known/reserved; warnings={len(registry.warnings)}')
    if registry.warnings:
        print('CAUTION: older PPO episode logs omit exact seeds; registry cannot prove complete history.')
    if mode == 'live':
        print(f'Team: {team_name} | credential source: {credential_source} (secret hidden)')
    answer = input('Type RUN to start this irreversible seed use: ').strip()
    if answer != 'RUN':
        print('Cancelled before using the seed.')
        return None
    reserve_seed(seed, run_id, mode, source_sha)
    command = [sys.executable, '-m', 'scripts.run_submission', str(config_path),
               '--report', str(official_report)]
    if mode == 'rehearsal':
        command.append('--dry-run')
    before = set(RESULTS.glob('*_submission.json'))
    with temporary_seed(seed, run_id):
        completed = subprocess.run(command, cwd=ROOT, env=env, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f'Official runner exited {completed.returncode}; seed is recorded as used. '
                           f'Check console and {official_report}')
    receipt = json.loads(official_report.read_text(encoding='utf-8'))
    score = receipt['score']
    if receipt['seed'] != seed or receipt['config']['scenario_number'] != 4:
        raise RuntimeError('Official receipt has an unexpected seed or scenario')
    submission = None
    digest = None
    if mode == 'live':
        created = sorted(set(RESULTS.glob('*_submission.json'))-before)
        if len(created) != 1:
            raise RuntimeError(f'Expected one new submission JSON, found {len(created)}')
        submission = created[0].resolve()
        digest = validate_submission(submission, seed, score, team_name)
    summary = dict(run_id=run_id, scenario=4, mode=mode, seed=seed, score=score,
                   agent_source=str(agent_source), agent_sha256=source_sha,
                   seed_registry_sha256=registry_sha, credential_source=credential_source,
                   official_receipt=str(official_report),
                   submission_json=str(submission) if submission else None,
                   submission_sha256=digest, uploaded=False)
    atomic_bytes(summary_path, (json.dumps(summary, indent=2)+'\n').encode('utf-8'))
    print('\n'+'='*62)
    print(f'QUALIFICATION RESULT | SEED {seed} | SCORE {score} POPS')
    print(f'Official receipt: {official_report}')
    if submission:
        print(f'UPLOAD THIS JSON: {submission}')
        print(f'JSON SHA256: {digest}')
        print('Contains team secret: do not show its contents or share publicly.')
    else:
        print('Rehearsal only; no submission JSON was created.')
    print(f'Clean result summary: {summary_path}')
    print('Scenario 4 seed parameter restored to its original bytes.')
    print('='*62)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, help='Organizer-provided seed; prompts if omitted')
    parser.add_argument('--mode', choices=('live', 'rehearsal'), help='Prompts if omitted')
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--credentials-file', type=Path)
    parser.add_argument('--recover', action='store_true', help='Restore an interrupted seed edit')
    parser.add_argument('--preflight', action='store_true', help='Check setup and seed without flying')
    args = parser.parse_args(argv)
    if args.recover:
        recover_seed_edit()
        return
    seed = args.seed
    if seed is None:
        raw = input('Seed from organizer (digits only): ').strip()
        if not raw.isdecimal():
            parser.error('Enter one nonnegative integer seed')
        seed = int(raw)
    if not 0 <= seed < 2**64:
        parser.error('Seed must be between 0 and 2^64-1')
    if args.preflight:
        verify_official_files()
        config, source = check_config(args.config)
        registry = collect(ROOT)
        status = 'USED/RESERVED - ASK FOR REDRAW' if seed in registry.origins else 'NOT FOUND IN LOCAL RECORD'
        print(f'Scenario 4 | {source.name} | seed {seed}: {status}')
        print(f'Known/reserved seeds: {len(registry.origins)}; coverage warnings: {len(registry.warnings)}')
        return
    mode = args.mode
    if mode is None:
        answer = input('Mode: type LIVE for real JSON, or press Enter for rehearsal: ').strip()
        mode = 'live' if answer == 'LIVE' else 'rehearsal' if not answer else None
        if mode is None:
            parser.error('Type LIVE or press Enter for rehearsal')
    run(seed, args.config, mode, args.credentials_file)


if __name__ == '__main__':
    try:
        main()
    except (ValueError, RuntimeError, FileNotFoundError, subprocess.CalledProcessError) as error:
        print(f'QUALIFICATION RUN STOPPED: {error}', file=sys.stderr)
        raise SystemExit(1) from None
