"""Audited, CPU-only route-data collection for the selected Scenario-4 agent.

The selected standalone source and official simulator are read-only. Each
seed runs in two fresh processes: a plain flight and a collector flight.
Action/observation hashes must match exactly before the observation-only
decision stream can be used as training data. The seed is provenance outside
all feature rows; no live qualification seed is inferred or loaded by an agent.
"""

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from zipfile import ZipFile, ZIP_DEFLATED

import yaml

from scripts.collect_route_shadow import paired_gate, update_hash


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / ('BalloonPoppingGymEnv/evaluation/configs/'
                         'submission_scenario4_guarded_innovation_launch42.yaml')
SEED_REGISTRY = ROOT / 'qualification/used_seeds.json'


def source_manifest(config_path):
    config_path = Path(config_path).resolve()
    if not config_path.is_relative_to(ROOT):
        raise ValueError('Config must be inside the workspace')
    config = yaml.safe_load(config_path.read_text(encoding='utf-8-sig'))
    if (config.get('scenario_number') != 4 or
            config.get('render_mode') is not None or
            config.get('leaderboard_submission') is not False or
            config.get('agent_class_name') != 'Scenario4GuardedInnovationAgent'):
        raise ValueError('Only the selected headless Scenario-4 agent is supported')
    source = (ROOT / config['agent_module_path']).resolve()
    agents = ROOT / 'BalloonPoppingGymEnv/agents'
    if (not source.is_relative_to(agents) or
            source.name != 'submission_scenario4_guarded_innovation_v1.py'):
        raise ValueError('Collector may not switch the reviewed agent source')
    files = (config_path, source, Path(__file__).resolve(),
             ROOT / 'BalloonPoppingGymEnv/training/route_shadow.py',
             ROOT / 'BalloonPoppingGymEnv/agents/route_attention_features.py',
             ROOT / 'scripts/collect_route_shadow.py',
             ROOT / 'scripts/audit_route_shadow.py',
             ROOT / 'scripts/diagnose_route_capture_ml.py')
    return config, {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in files}


def flight(config_path, output, seed, collect):
    from BalloonPoppingGymEnv.agents.base_agent import BaseAgent
    from BalloonPoppingGymEnv.envs.balloon_world import BalloonPoppingEnv
    from BalloonPoppingGymEnv.evaluation.evaluate import (
        _load_agent_class, load_scenario_parameters,
    )
    from BalloonPoppingGymEnv.training.route_shadow import RouteShadowCollector

    config, manifest = source_manifest(config_path)
    cls = _load_agent_class(config['agent_module_path'],
                            config['agent_class_name'])
    if not issubclass(cls, BaseAgent):
        raise ValueError('Invalid agent class')
    parameters, given = load_scenario_parameters(4)
    parameters = copy.deepcopy(parameters)
    parameters['scenario']['random_seed'] = seed
    agent = cls(given, **config.get('agent_kwargs', {}))
    env = BalloonPoppingEnv(render_mode=None, parameters=parameters)
    actions, observations = hashlib.sha256(), hashlib.sha256()
    writer = None
    collector = None
    start = time.monotonic()
    try:
        if collect:
            writer = (output / 'decisions.jsonl').open('x', encoding='utf-8')
            def emit(row):
                writer.write(json.dumps(row, allow_nan=False,
                                        separators=(',', ':')) + '\n')
            collector = RouteShadowCollector(emit)
        observation, info = env.reset(seed=seed)
        update_hash(observations, observation)
        if collector is not None:
            collector.observe(observation)
        steps = 0
        terminated = truncated = False
        while not (terminated or truncated):
            action = agent.get_action(observation)
            update_hash(actions, action)
            if collector is not None:
                collector.after_action(agent, observation)
            observation, _, terminated, truncated, info = env.step(action)
            update_hash(observations, observation)
            if collector is not None:
                collector.observe(observation)
            steps += 1
        counts = collector.finish(observation) if collector else None
        if counts is not None and (counts['initial_pops'] +
                                   counts['observed_pops'] !=
                                   int(info['popped_count'])):
            raise RuntimeError('Observed pop stream disagrees with official score')
        if any(hashlib.sha256(Path(name).read_bytes()).hexdigest() != digest
               for name, digest in manifest.items()):
            raise RuntimeError('Collection source changed during flight')
        result = dict(score=int(info['popped_count']), steps=steps,
                      final_time=float(observation['simulation_time']),
                      terminated=bool(terminated), truncated=bool(truncated),
                      action_sha256=actions.hexdigest(),
                      observation_sha256=observations.hexdigest(),
                      collection_counts=counts,
                      wall_seconds=time.monotonic() - start,
                      source_manifest=manifest)
        label = 'shadow' if collect else 'baseline'
        (output / f'{label}.json').write_text(
            json.dumps(result, indent=2), encoding='utf-8')
        print(json.dumps(dict(label=label, seed=seed, score=result['score'],
                              steps=steps, wall_seconds=result['wall_seconds'])),
              flush=True)
    finally:
        if writer is not None:
            writer.close()
        env.close()


def validate_training_seeds(seeds, allow_recorded=False):
    if (not seeds or len(seeds) != len(set(seeds)) or
            any(type(seed) is not int or not 0 <= seed < 1_000_000_000
                for seed in seeds)):
        raise ValueError('Unique training-pool seeds [0, 1e9) required')
    if not allow_recorded:
        registry = json.loads(SEED_REGISTRY.read_text(encoding='utf-8'))['seeds']
        overlap = [seed for seed in seeds if str(seed) in registry]
        if overlap:
            raise ValueError(f'Seed already recorded: {overlap}')


def run(config_path, output, seeds, allow_recorded=False):
    validate_training_seeds(seeds, allow_recorded)
    config, manifest = source_manifest(config_path)
    output = Path(output).resolve()
    if not output.is_relative_to(ROOT):
        raise ValueError('Output must be inside the workspace')
    output.mkdir(parents=True, exist_ok=False)
    (output / 'plan.json').write_text(json.dumps(dict(
        scenario=4, seeds=seeds, purpose='selected_guarded_route_training_data',
        controller=config['agent_module_path'], source_manifest=manifest,
        exact_action_observation_parity=True, official_simulator_fresh=True,
        device='cpu', workers=1, uploaded=False), indent=2),
        encoding='utf-8')
    child_env = dict(os.environ, CUDA_VISIBLE_DEVICES='',
                     OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                     MKL_NUM_THREADS='1',
                     PYTHONPATH=os.pathsep.join((str(ROOT / 'ActiveRocketPy'),
                                                 str(ROOT))))
    completed = []
    try:
        for seed in seeds:
            if any(hashlib.sha256(Path(name).read_bytes()).hexdigest() != digest
                   for name, digest in manifest.items()):
                raise RuntimeError('Frozen source changed before seed')
            folder = output / str(seed)
            folder.mkdir()
            for label in ('baseline', 'shadow'):
                command = (sys.executable, '-m',
                           'scripts.collect_guarded_route_shadow',
                           '--config', str(config_path), '--output', str(folder),
                           '--seeds', str(seed), '--flight', label)
                with (folder / f'{label}.log').open('x', encoding='utf-8') as log:
                    subprocess.run(command, cwd=ROOT, env=child_env, stdout=log,
                                   stderr=subprocess.STDOUT, check=True, timeout=1500)
            pair = {label: json.loads((folder / f'{label}.json').read_text())
                    for label in ('baseline', 'shadow')}
            if not paired_gate(pair['baseline'], pair['shadow']):
                raise RuntimeError(f'Action/observation parity failed on {seed}')
            from scripts.audit_route_shadow import audit
            gate = dict(seed=seed, exact_parity=True,
                        dataset_sha256=hashlib.sha256(
                            (folder / 'decisions.jsonl').read_bytes()).hexdigest(),
                        **pair)
            (folder / 'gate.json').write_text(json.dumps(gate, indent=2),
                                              encoding='utf-8')
            audited = audit(folder)
            if not audited['all_pop_events_complete']:
                raise RuntimeError(f'Incomplete pop event audit on {seed}')
            (folder / 'audit.json').write_text(json.dumps(audited, indent=2),
                                               encoding='utf-8')
            completed.append(dict(seed=seed, score=pair['shadow']['score'],
                                  attempts=audited['executed_attempts'],
                                  audit_sha256=hashlib.sha256(
                                      (folder / 'audit.json').read_bytes()).hexdigest()))
            (output / 'progress.json').write_text(json.dumps(dict(
                completed=completed, completed_pairs=len(completed),
                planned_pairs=len(seeds)), indent=2), encoding='utf-8')
            print(f'PARITY PASS seed={seed} score={pair["shadow"]["score"]} '
                  f'attempts={audited["executed_attempts"]}', flush=True)
        (output / 'summary.json').write_text(json.dumps(dict(
            completed=completed, all_passed=True), indent=2), encoding='utf-8')
        archive = output.with_suffix('.zip')
        if archive.exists():
            raise FileExistsError(f'Archive already exists: {archive}')
        with ZipFile(archive, 'x', compression=ZIP_DEFLATED) as zip_file:
            for path in sorted(output.rglob('*')):
                if path.is_file():
                    zip_file.write(path, 'data/' + str(path.relative_to(output)).replace('\\', '/'))
        if len(seeds) >= 3:
            from scripts.diagnose_route_capture_ml import (
                fit_regularized_logistic, held_out_report,
                load_audited_attempts,
            )
            import numpy as np
            records = load_audited_attempts(archive)
            report = held_out_report(records)
            model = fit_regularized_logistic(
                np.asarray([row['features'] for row in records]),
                np.asarray([row['hit'] for row in records]))
            report['full_fit_prototype'] = dict(
                center=model[0].tolist(), scale=model[1].tolist(),
                coefficients=model[2].tolist(),
                role='offline prototype; not connected to the competition agent')
            report['source_archive_sha256'] = hashlib.sha256(
                archive.read_bytes()).hexdigest()
            (output / 'route_capture_fit.json').write_text(
                json.dumps(report, indent=2), encoding='utf-8')
            print('OFFLINE_FIT ' + json.dumps(dict(
                fields=report['fields'], attempts=report['attempts'],
                hits=report['hits'], prior=report['prior'],
                learned=report['learned'])), flush=True)
        print(f'COMPLETE pairs={len(completed)} archive={archive}', flush=True)
    except Exception as exc:
        (output / 'failed.json').write_text(json.dumps(dict(
            error=str(exc), completed_pairs=len(completed)), indent=2),
            encoding='utf-8')
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seeds', type=int, nargs='+', required=True)
    parser.add_argument('--allow-recorded', action='store_true')
    parser.add_argument('--flight', choices=('baseline', 'shadow'),
                        help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.flight:
        if len(args.seeds) != 1:
            parser.error('One seed per child flight')
        flight(args.config, args.output, args.seeds[0],
               args.flight == 'shadow')
    else:
        run(args.config, args.output, args.seeds, args.allow_recorded)


if __name__ == '__main__':
    main()
