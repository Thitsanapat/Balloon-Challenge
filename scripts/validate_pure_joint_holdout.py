"""Serial, fresh-process paired Scenario-4 test of pure joint refresh.

Seeds and promotion screen were frozen before either policy saw these fields.
The job waits for the in-progress audited training-pool recovery to finish so
it does not compete with its simulations. It never changes the live launcher.
"""

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from scripts.qualification_run import ROOT


SEEDS = (605131410, 435857031, 927297526, 298628555)
RESULTS = ROOT / 'BalloonPoppingGymEnv/evaluation/results'
PREVIOUS = RESULTS / 'pure_joint_holdout4_20261009'
OUTPUT = RESULTS / 'pure_joint_holdout4_recovered_20261009'
RECOVERY = RESULTS / 'guarded_route_train12_b_recovered_20261009'
CONFIGS = dict(
    selected=ROOT / 'BalloonPoppingGymEnv/evaluation/configs/submission_scenario4_guarded_innovation_launch42.yaml',
    candidate=ROOT / 'BalloonPoppingGymEnv/evaluation/configs/scenario4_pure_joint_refresh_launch42.yaml',
)
SOURCES = (
    ROOT / 'BalloonPoppingGymEnv/agents/submission_scenario4_guarded_innovation_v1.py',
    ROOT / 'BalloonPoppingGymEnv/agents/scenario4_pure_joint_refresh_agent.py',
)


def fingerprints():
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (*CONFIGS.values(), *SOURCES)}


def wait_for_recovery(max_seconds=8 * 3600):
    deadline = time.monotonic() + max_seconds
    while time.monotonic() < deadline:
        if (RECOVERY / 'failed.json').exists():
            raise RuntimeError('Audited data recovery failed; paired test not started')
        if ((RECOVERY / 'route_capture_fit.json').exists() and
                RECOVERY.with_suffix('.zip').exists()):
            summary = json.loads((RECOVERY / 'summary.json').read_text())
            if summary.get('all_passed') and len(summary.get('completed', ())) == 12:
                return
            raise RuntimeError('Incomplete recovery summary')
        time.sleep(30)
    raise TimeoutError('Audited data recovery did not finish within eight hours')


def read_single_flight(path):
    """evaluate_guidance serializes even one seed as a one-element list."""
    value = json.loads(Path(path).read_text())
    if not isinstance(value, list) or len(value) != 1 or not isinstance(value[0], dict):
        raise ValueError(f'Expected one official flight in {path}')
    return value[0]


def run():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    registry = json.loads((ROOT / 'qualification/used_seeds.json').read_text())
    manual = (ROOT / 'qualification/used_seed_manual.jsonl').read_text()
    if any(str(seed) not in registry['seeds'] or
           f'"seed": {seed}, "usage": "scenario4_pure_joint_refresh_paired_holdout_predeclared_20261009"'
           not in manual for seed in SEEDS):
        raise ValueError('Holdout seeds were not recorded before testing')
    frozen = fingerprints()
    old_plan = json.loads((PREVIOUS / 'plan.json').read_text())
    if (not (PREVIOUS / 'failed.json').exists() or
            tuple(old_plan['seeds']) != SEEDS or
            old_plan['source_sha256'] != frozen or
            old_plan['gate'] != dict(min_total_gain=2,
                                    max_single_regression=1,
                                    min_candidate_wins=2)):
        raise ValueError('Previous incomplete run differs from frozen plan')
    old_first = read_single_flight(
        PREVIOUS / f'{SEEDS[0]}_selected.json')
    if (old_first['seed'] != SEEDS[0] or old_first['scenario'] != 4 or
            old_first['agent_sha256'] != frozen[str(SOURCES[0].relative_to(ROOT))] or
            not old_first['terminated'] or old_first['truncated'] or
            not old_first['agent_file_unchanged_during_run'] or
            not old_first['agent_sources_unchanged_during_run']):
        raise ValueError('Previous selected flight did not pass recovery audit')
    OUTPUT.mkdir()
    (OUTPUT / 'plan.json').write_text(json.dumps(dict(
        scenario=4, seeds=SEEDS, order=('selected', 'candidate'),
        source_sha256=frozen, official_fresh_process_per_flight=True,
        auto_promote=False,
        recovered_from=str(PREVIOUS), copied_selected_flight_seed=SEEDS[0],
        gate=dict(min_total_gain=2, max_single_regression=1,
                  min_candidate_wins=2)), indent=2), encoding='utf-8')
    for suffix in ('json', 'log'):
        name = f'{SEEDS[0]}_selected.{suffix}'
        shutil.copy2(PREVIOUS / name, OUTPUT / name)
    wait_for_recovery()
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='',
               OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               MKL_NUM_THREADS='1',
               PYTHONPATH=os.pathsep.join((str(ROOT / 'ActiveRocketPy'),
                                           str(ROOT))))
    pairs = []
    try:
        for seed in SEEDS:
            result = {'seed': seed}
            for policy in ('selected', 'candidate'):
                if fingerprints() != frozen:
                    raise RuntimeError('Frozen source/config changed between flights')
                report = OUTPUT / f'{seed}_{policy}.json'
                if not report.exists():
                    command = (sys.executable, '-m', 'scripts.evaluate_guidance',
                               str(CONFIGS[policy]), '--seeds', str(seed),
                               '--report', str(report))
                    with (OUTPUT / f'{seed}_{policy}.log').open(
                            'x', encoding='utf-8') as log:
                        subprocess.run(command, cwd=ROOT, env=env, stdout=log,
                                       stderr=subprocess.STDOUT, check=True,
                                       timeout=1800)
                flight = read_single_flight(report)
                if (flight['seed'] != seed or flight['scenario'] != 4 or
                        not flight['terminated'] or flight['truncated'] or
                        not flight['agent_file_unchanged_during_run'] or
                        not flight['agent_sources_unchanged_during_run']):
                    raise RuntimeError(f'Incomplete or changed flight {seed}/{policy}')
                result[policy] = dict(
                    score=int(flight['score']),
                    agent_sha256=flight['agent_sha256'],
                    pop_events=flight['pop_events'])
            pairs.append(result)
            (OUTPUT / 'progress.json').write_text(json.dumps(dict(
                completed_pairs=len(pairs), planned_pairs=len(SEEDS),
                pairs=pairs), indent=2), encoding='utf-8')
            print(f'PAIR seed={seed} selected={result["selected"]["score"]} '
                  f'candidate={result["candidate"]["score"]}', flush=True)
        selected = [item['selected']['score'] for item in pairs]
        candidate = [item['candidate']['score'] for item in pairs]
        differences = [new - old for new, old in zip(candidate, selected)]
        passed = (sum(differences) >= 2 and min(differences) >= -1 and
                  sum(delta > 0 for delta in differences) >= 2)
        summary = dict(seeds=SEEDS, selected=selected, candidate=candidate,
                       differences=differences,
                       selected_total=sum(selected),
                       candidate_total=sum(candidate), gate_passed=passed,
                       live_agent_changed=False)
        (OUTPUT / 'summary.json').write_text(
            json.dumps(summary, indent=2), encoding='utf-8')
        print(json.dumps(summary), flush=True)
    except Exception as exc:
        (OUTPUT / 'failed.json').write_text(json.dumps(dict(
            error=str(exc), completed_pairs=len(pairs)), indent=2),
            encoding='utf-8')
        raise


if __name__ == '__main__':
    run()
