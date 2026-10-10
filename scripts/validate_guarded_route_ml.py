"""Wait for the frozen route-ML fit, then run a predeclared paired holdout.

This never modifies the selected qualification agent or submission config.
It stops without flying the reserved fields if the model calibration gate fails.
Only one official simulator flight runs at a time, with no GPU requested.
"""

import hashlib
import json
import os
from pathlib import Path
import time

import yaml

from scripts.evaluate_guidance import evaluate
from scripts.package_guarded_capture_model import package
from scripts.qualification_run import ROOT, verify_official_files
from scripts.validate_scenario4_acceleration import atomic_json


SEEDS = (761692544, 234703015, 439862696, 130752622)
DATA = ROOT / 'BalloonPoppingGymEnv/evaluation/results/guarded_route_train12_20261008'
RESULT = ROOT / 'BalloonPoppingGymEnv/evaluation/results/guarded_route_ml_holdout4_20261008.json'
CONFIGS = {
    'selected': ROOT / 'BalloonPoppingGymEnv/evaluation/configs/submission_scenario4_guarded_innovation_launch42.yaml',
    'learned': ROOT / 'BalloonPoppingGymEnv/evaluation/configs/scenario4_learned_route_launch42.yaml',
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def wait_for_fit(max_seconds=18000):
    deadline = time.monotonic() + max_seconds
    while time.monotonic() < deadline:
        if (DATA / 'failed.json').exists():
            raise RuntimeError('Training collection failed; inspect failed.json')
        if (DATA / 'route_capture_fit.json').exists():
            summary = json.loads((DATA / 'summary.json').read_text())
            if not summary.get('all_passed') or len(summary['completed']) != 12:
                raise RuntimeError('Incomplete paired training archive')
            return
        time.sleep(30)
    raise TimeoutError('Training collection did not finish within five hours')


def expected_plan():
    configs = {}
    for name, path in CONFIGS.items():
        config = yaml.safe_load(path.read_text(encoding='utf-8-sig'))
        if config['scenario_number'] != 4 or config['render_mode'] is not None:
            raise ValueError('Only headless Scenario 4 configs accepted')
        source = (ROOT / config['agent_module_path']).resolve()
        configs[name] = dict(config_path=str(path), config_sha256=sha(path),
                             agent_source=str(source), agent_sha256=sha(source))
    return dict(schema='guarded_route_ml_holdout_v1', seeds=list(SEEDS),
                order=list(CONFIGS), configs=configs,
                selected_dependency_sha256=sha(
                    ROOT / 'BalloonPoppingGymEnv/agents/submission_scenario4_guarded_innovation_v1.py'),
                model_sha256=sha(DATA / 'capture_weights.json'))


def run():
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    wait_for_fit()
    try:
        package(DATA / 'route_capture_fit.json', DATA / 'capture_weights.json')
    except ValueError as exc:
        atomic_json(RESULT, dict(complete=True, model_rejected=True,
                                 reason=str(exc), held_out_flights=0))
        print(f'MODEL REJECTED: {exc}', flush=True)
        return
    verify_official_files()
    plan = expected_plan()
    if RESULT.exists():
        data = json.loads(RESULT.read_text(encoding='utf-8'))
        if data.get('plan') != plan:
            raise RuntimeError('Frozen holdout plan or source changed')
    else:
        data = dict(plan=plan, flights=[], complete=False)
        atomic_json(RESULT, data)
    seen = {(item['seed'], item['policy']) for item in data['flights']}
    for seed in SEEDS:
        for name, path in CONFIGS.items():
            if (seed, name) in seen:
                continue
            if expected_plan() != plan:
                raise RuntimeError('Holdout source/config/model changed before flight')
            config = yaml.safe_load(path.read_text(encoding='utf-8-sig'))
            print(f'START seed={seed} policy={name}', flush=True)
            result = evaluate(config, seed)
            if (result['seed'] != seed or result['scenario'] != 4 or
                    result['agent_sha256'] != plan['configs'][name]['agent_sha256'] or
                    not result['terminated'] or result['truncated'] or
                    not result['agent_sources_unchanged_during_run'] or
                    expected_plan() != plan):
                raise RuntimeError(f'Official/source gate failed on {seed}/{name}')
            data['flights'].append(dict(seed=seed, policy=name, result=result))
            seen.add((seed, name))
            atomic_json(RESULT, data)
            print(f'COMPLETE seed={seed} policy={name} score={result["score"]}',
                  flush=True)
    scores = {name: [next(item['result']['score'] for item in data['flights']
                          if item['seed'] == seed and item['policy'] == name)
                     for seed in SEEDS] for name in CONFIGS}
    data.update(complete=True, model_rejected=False, scores=scores,
                totals={name: sum(values) for name, values in scores.items()},
                selected_qualification_agent_unchanged=True,
                automatic_promotion=False)
    atomic_json(RESULT, data)
    print(f'HOLDOUT COMPLETE totals={data["totals"]} result={RESULT}',
          flush=True)


if __name__ == '__main__':
    run()
