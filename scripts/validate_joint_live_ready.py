"""Explicit LOCAL packaging rehearsal; never uploads and never uses organizer seeds.

Use a fresh development seed. The standard runner records it before flight,
rejects reuse, checks the official simulator and restores the scenario file.
"""
import argparse
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

from scripts import qualification_run as runner

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'BalloonPoppingGymEnv/evaluation/configs/submission_scenario4_pure_joint_launch42.yaml'
EXPECTED = '9ff7f93d21a945d0479289be57bd893d8f33ab169fb1a6bfeb153ea442b6efb9'
REPORT = ROOT / 'qualification/finalnight_joint_readiness.json'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--confirm-local-packaging', action='store_true', required=True)
    args = parser.parse_args()
    if REPORT.exists():
        raise FileExistsError('Readiness report already exists; do not repeat this seed')
    _, source = runner.check_config(CONFIG)
    if runner.sha256_file(source) != EXPECTED:
        raise ValueError('Not the previously parity-checked standalone')
    original = runner.SCENARIO_FILE.read_bytes()
    print('LOCAL PACKAGING REHEARSAL ONLY. No server upload. Not an organizer evaluation.', flush=True)
    # The CLI flag explicitly authorizes this local rehearsal. All reuse checks,
    # seed reservation and official-file validation remain in the real runner.
    with patch('builtins.input', return_value='RUN'):
        result = runner.run(args.seed, CONFIG, 'live')
    if runner.SCENARIO_FILE.read_bytes() != original or runner.ACTIVE.exists():
        raise RuntimeError('Scenario restoration failed')
    if result['agent_sha256'] != EXPECTED or result['uploaded'] or not result['submission_json']:
        raise RuntimeError('Unexpected packaged result')
    report = dict(purpose='local rehearsal, not a live organizer evaluation',
                  passed=True, scenario_restored=True, result=result,
                  scenario_sha256=hashlib.sha256(original).hexdigest())
    with REPORT.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2)
    print(f'LOCAL READINESS CHECK PASSED: {REPORT}', flush=True)


if __name__ == '__main__':
    main()
