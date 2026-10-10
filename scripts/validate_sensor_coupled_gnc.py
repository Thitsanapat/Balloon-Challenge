"""Two-field DEVELOPMENT ablation of guidance, navigation and control.

All policies run in fresh official simulator processes, at most two CPU
workers and no visible GPU. This does not upload or change the live launcher.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import yaml


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'BalloonPoppingGymEnv/evaluation/results'
OUTPUT = RESULTS / 'sensor_coupled_gnc_dev_20261009'
PENDING = RESULTS / 'pure_joint_holdout4_recovered_20261009'
SEEDS = (2, 0)
MODES = dict(joint=(False, False), navigation=(True, False),
             control=(False, True), combined=(True, True))


def read_flight(path):
    rows = json.loads(path.read_text())
    if not isinstance(rows, list) or len(rows) != 1:
        raise ValueError(f'Expected exactly one flight: {path}')
    row = rows[0]
    if (row['scenario'] != 4 or not row['terminated'] or row['truncated'] or
            not row['agent_file_unchanged_during_run'] or
            not row['agent_sources_unchanged_during_run']):
        raise ValueError(f'Incomplete or modified flight: {path}')
    return row


def summarize(row):
    diag = row['diagnostics']
    episodes = row.get('guidance_telemetry', {}).get('target_episodes', [])
    misses = [ep['true_closest'] for ep in episodes
              if ep.get('end_reason') != 'popped' and ep.get('true_closest')]
    return dict(score=row['score'], navigation=row['navigation_error'],
                mean_tracking_error=diag['position_error_sum'] /
                max(1, diag['tracking_steps']),
                saturation_steps=diag['saturated_steps'],
                missed_targets_within_3m=sum(m['distance'] < 3.0 for m in misses),
                angular_response_gain=diag.get('angular_response_gain'),
                angular_response_samples=diag.get('angular_response_samples', 0),
                adaptive_control_steps=diag.get('adaptive_control_steps', 0),
                post_reset_wall_seconds=row['post_reset_wall_seconds'])


def run():
    registry = json.loads((ROOT / 'qualification/used_seeds.json').read_text())
    if any(str(seed) not in registry['seeds'] for seed in SEEDS):
        raise ValueError('Development seeds must already be recorded')
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    OUTPUT.mkdir()
    paths = sorted((ROOT / 'BalloonPoppingGymEnv/agents').glob('*.py'))
    paths += [ROOT / 'scripts/evaluate_guidance.py',
              ROOT / 'scripts/guidance_telemetry.py']
    frozen = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in paths}
    plan = dict(seeds=SEEDS, scenario=4, stage='development, not held-out',
                modes=MODES, source_sha256=frozen, cpu_workers=2,
                gpu_visible=False, auto_promote=False,
                screen='total gain >=2 over joint, no seed regression >1')
    (OUTPUT / 'plan.json').write_text(json.dumps(plan, indent=2), encoding='utf-8')
    deadline = time.monotonic() + 3600
    while not (PENDING / 'summary.json').exists():
        if (PENDING / 'failed.json').exists() or time.monotonic() > deadline:
            raise RuntimeError('Prior holdout did not finish successfully')
        time.sleep(15)
    base = yaml.safe_load((ROOT / 'BalloonPoppingGymEnv/evaluation/configs/'
                           'scenario4_pure_joint_refresh_launch42.yaml').read_text())
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1',
               OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
               PYTHONPATH=os.pathsep.join((str(ROOT / 'ActiveRocketPy'), str(ROOT))))

    def flight(seed, mode):
        for name, digest in frozen.items():
            if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest:
                raise RuntimeError(f'Frozen source changed: {name}')
        config = dict(base, agent_kwargs=dict(base['agent_kwargs']),
                      record_guidance_telemetry=True)
        if mode != 'joint':
            config['agent_module_path'] = (
                './BalloonPoppingGymEnv/agents/scenario4_sensor_coupled_agent.py')
            config['agent_class_name'] = 'Scenario4SensorCoupledAgent'
            config['agent_name'] = f'Sensor coupled GNC development {mode}'
            config['agent_kwargs'].update(
                navigation_trapezoid=MODES[mode][0],
                adaptive_angular_response=MODES[mode][1])
        config_path = OUTPUT / f'{seed}_{mode}.yaml'
        config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')
        report = OUTPUT / f'{seed}_{mode}.json'
        with (OUTPUT / f'{seed}_{mode}.log').open('x', encoding='utf-8') as log:
            subprocess.run((sys.executable, '-m', 'scripts.evaluate_guidance',
                            str(config_path), '--seeds', str(seed),
                            '--report', str(report)), cwd=ROOT, env=env,
                           stdout=log, stderr=subprocess.STDOUT,
                           check=True, timeout=1800)
        row = read_flight(report)
        if row['seed'] != seed:
            raise ValueError('Seed mismatch')
        return seed, mode, summarize(row)

    results = {}
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            jobs = [pool.submit(flight, seed, mode) for seed in SEEDS for mode in MODES]
            for future in as_completed(jobs):
                seed, mode, result = future.result()
                results.setdefault(str(seed), {})[mode] = result
                (OUTPUT / 'progress.json').write_text(
                    json.dumps(results, indent=2), encoding='utf-8')
                print(f'COMPLETE seed={seed} mode={mode} score={result["score"]}', flush=True)
        screens = {}
        for mode in ('navigation', 'control', 'combined'):
            delta = [results[str(seed)][mode]['score'] -
                     results[str(seed)]['joint']['score'] for seed in SEEDS]
            screens[mode] = dict(deltas=delta, total_gain=sum(delta),
                                 passed=sum(delta) >= 2 and min(delta) >= -1)
        (OUTPUT / 'summary.json').write_text(json.dumps(dict(
            stage='development only; independent validation still needed',
            results=results, screens=screens, live_agent_changed=False), indent=2),
            encoding='utf-8')
        print(json.dumps(screens), flush=True)
    except Exception as exc:
        (OUTPUT / 'failed.json').write_text(json.dumps(dict(error=str(exc))), encoding='utf-8')
        raise


if __name__ == '__main__':
    run()
