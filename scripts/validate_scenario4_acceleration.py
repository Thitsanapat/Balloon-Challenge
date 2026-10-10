"""Predeclared paired Scenario-4 holdout for the short-horizon acceleration agent.

Each seed gets two fresh official BalloonPoppingEnv resets, with no cached
balloon field. Results are checkpointed after each flight. No submission JSON
or team credential is involved. The four seeds and agent files are frozen
before any holdout score is inspected.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile

import yaml

from scripts.evaluate_guidance import evaluate
from scripts.qualification_run import ROOT, verify_official_files


SEEDS = (1652635469, 1504633979, 1154366829, 1630657292)
CONFIGS = {
    'reviewed': ROOT/'BalloonPoppingGymEnv/evaluation/configs/submission_scenario4_wind_profile_launch42.yaml',
    'acceleration_active_h3': ROOT/'BalloonPoppingGymEnv/evaluation/configs/scenario4_acceleration_wind_launch42.yaml',
}
RESULT = ROOT/'BalloonPoppingGymEnv/evaluation/results/s4_acceleration_holdout4_20261007.json'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile('w', encoding='utf-8', dir=path.parent,
                                     prefix='.acceleration_holdout_', suffix='.tmp',
                                     delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
    temporary.replace(path)


def expected_plan():
    configs = {}
    for name, path in CONFIGS.items():
        config = yaml.safe_load(path.read_text(encoding='utf-8-sig'))
        if config['scenario_number'] != 4:
            raise ValueError('Expected only Scenario 4 configs')
        source = (ROOT/config['agent_module_path']).resolve()
        configs[name] = dict(config_path=str(path), config_sha256=sha(path),
                             agent_source=str(source), agent_sha256=sha(source))
    return dict(schema='scenario4_acceleration_holdout_v1',
                seeds=list(SEEDS), order=list(CONFIGS), configs=configs)


def run():
    verify_official_files()
    plan = expected_plan()
    if RESULT.exists():
        data = json.loads(RESULT.read_text(encoding='utf-8'))
        if data['plan'] != plan:
            raise RuntimeError('Saved plan or source changed; refusing unsafe resume')
    else:
        data = dict(plan=plan, flights=[], complete=False)
        atomic_json(RESULT, data)
    seen = {(item['seed'], item['policy']) for item in data['flights']}
    for seed in SEEDS:
        for name, path in CONFIGS.items():
            if (seed, name) in seen:
                continue
            if expected_plan() != plan:
                raise RuntimeError('Holdout source/config changed before flight')
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
    data['complete'] = True
    scores = {name: [next(item['result']['score'] for item in data['flights']
                          if item['seed'] == seed and item['policy'] == name)
                     for seed in SEEDS] for name in CONFIGS}
    data['scores'] = scores
    data['totals'] = {name: sum(values) for name, values in scores.items()}
    atomic_json(RESULT, data)
    print(f'HOLDOUT COMPLETE totals={data["totals"]} result={RESULT}', flush=True)


if __name__ == '__main__':
    run()
