"""CPU-only paired flight parity gate and observation-only route collection.

No official simulator/evaluator or frozen controller changes. Each flight runs
in a fresh process. Dataset seeds are from the training pool, NOT validation or
holdout. Seed stays in external provenance, never in a decision feature row.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in sorted(value.items())}
    if isinstance(value, (tuple, list)):
        return [jsonable(v) for v in value]
    if hasattr(value, 'tolist'):
        return jsonable(value.tolist())
    # Observation NaN is legitimate before ignition; canonicalize for hashing.
    if isinstance(value, float):
        import math
        if not math.isfinite(value):
            return 'nan' if math.isnan(value) else ('+inf' if value > 0 else '-inf')
    return value


def update_hash(hasher, value):
    hasher.update(json.dumps(jsonable(value), sort_keys=True, separators=(',', ':'),
                             allow_nan=False).encode('utf-8') + b'\n')


def paired_gate(baseline, shadow):
    keys = ('action_sha256', 'observation_sha256', 'steps', 'score', 'final_time',
            'terminated', 'truncated')
    return (all(baseline[k] == shadow[k] for k in keys)
            and baseline['terminated'] and not baseline['truncated'])


def flight(checkpoint, output, seed, collect):
    import copy
    from BalloonPoppingGymEnv.envs.balloon_world import BalloonPoppingEnv
    from BalloonPoppingGymEnv.evaluation.evaluate import load_scenario_parameters
    from BalloonPoppingGymEnv.agents.multiscenario_ppo_policy_agent import MultiScenarioPPOPolicyAgent
    from BalloonPoppingGymEnv.training.route_shadow import RouteShadowCollector
    parameters, given = load_scenario_parameters(4)
    parameters = copy.deepcopy(parameters)
    parameters['scenario']['random_seed'] = seed
    agent = MultiScenarioPPOPolicyAgent(given, checkpoint=str(checkpoint))
    env = BalloonPoppingEnv(render_mode=None, parameters=parameters)
    actions, observations = hashlib.sha256(), hashlib.sha256()
    writer = None
    observer = None
    start = time.monotonic()
    try:
        if collect:
            writer = (output/'decisions.jsonl').open('x', encoding='utf-8')
            def emit(row):
                writer.write(json.dumps(row, allow_nan=False, separators=(',', ':')) + '\n')
            observer = RouteShadowCollector(emit)
        obs, info = env.reset(seed=seed)
        update_hash(observations, obs)
        if observer is not None:
            observer.observe(obs)
        steps = 0
        terminated = truncated = False
        while not (terminated or truncated):
            action = agent.get_action(obs)
            update_hash(actions, action)
            if observer is not None:
                observer.after_action(agent, obs)
            obs, _, terminated, truncated, info = env.step(action)
            update_hash(observations, obs)
            if observer is not None:
                observer.observe(obs)
            steps += 1
        counts = observer.finish(obs) if observer is not None else None
        if counts is not None and counts['initial_pops'] + counts['observed_pops'] != int(info['popped_count']):
            raise RuntimeError('Observed pop transitions disagree with official score')
        result = dict(score=int(info['popped_count']), steps=steps,
                      final_time=float(obs['simulation_time']), terminated=bool(terminated),
                      truncated=bool(truncated), action_sha256=actions.hexdigest(),
                      observation_sha256=observations.hexdigest(),
                      collection_counts=counts, wall_seconds=time.monotonic()-start)
        label = 'shadow' if collect else 'baseline'
        (output/f'{label}.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
        print(json.dumps(dict(label=label, **result)), flush=True)
    finally:
        if writer is not None:
            writer.close()
        env.close()


def run(checkpoint, output, seeds):
    if (not seeds or len(set(seeds)) != len(seeds)
            or any(not 0 <= s < 1000000000 for s in seeds)):
        raise ValueError('Unique training-pool seeds in [0, 1000000000) required')
    checkpoint, output = Path(checkpoint).resolve(), Path(output).resolve()
    manifest_path = checkpoint.parent/'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    files = {ROOT/name: sha for name, sha in manifest['source_sha256'].items()}
    bundle = ROOT/'bundle_manifest.json'
    if bundle.exists():
        for name, sha in json.loads(bundle.read_text())['files'].items():
            path = ROOT/name
            if path in files and files[path] != sha:
                raise ValueError(f'Conflicting source manifests: {name}')
            files[path] = sha
    for path, sha in files.items():
        if digest(path) != sha:
            raise ValueError(f'Frozen source mismatch: {path}')
    for path in [checkpoint, manifest_path, Path(__file__),
                 ROOT/'scripts/audit_route_shadow.py',
                 ROOT/'BalloonPoppingGymEnv/training/route_shadow.py',
                 ROOT/'BalloonPoppingGymEnv/agents/route_attention_features.py',
                 ROOT/'BalloonPoppingGymEnv/agents/multiscenario_ppo_policy_agent.py']:
        files[path] = digest(path)
    def check_frozen():
        if any(digest(path) != sha for path, sha in files.items()):
            raise RuntimeError('Source/checkpoint changed during paired collection')
    output.mkdir(parents=True, exist_ok=False)
    plan = dict(scenario=4, seeds=seeds, purpose='training_data_and_shadow_parity',
                event_schema='route_shadow_events_v2', audit_each_pair=True,
                frozen_sha256={str(p): sha for p, sha in files.items()},
                device='cpu', workers=1, timeout_seconds_per_flight=1500,
                source_of_labels='subsequent_permitted_observation', uploaded=False)
    (output/'plan.json').write_text(json.dumps(plan, indent=2), encoding='utf-8')
    child_env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1',
                     OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
                     PYTHONPATH=os.pathsep.join([str(ROOT/'ActiveRocketPy'), str(ROOT)]))
    completed = []
    try:
        for seed in seeds:
            check_frozen()
            directory = output/str(seed)
            directory.mkdir()
            for label in ('baseline', 'shadow'):
                check_frozen()
                command = [sys.executable, '-m', 'scripts.collect_route_shadow',
                           '--checkpoint', str(checkpoint), '--output', str(directory),
                           '--seeds', str(seed), '--flight', label]
                with (directory/f'{label}.log').open('x', encoding='utf-8') as log:
                    subprocess.run(command, cwd=ROOT, env=child_env, stdout=log,
                                   stderr=subprocess.STDOUT, check=True, timeout=1500)
                check_frozen()
            pair = {k: json.loads((directory/f'{k}.json').read_text()) for k in ('baseline', 'shadow')}
            passed = paired_gate(pair['baseline'], pair['shadow'])
            row = dict(seed=seed, exact_parity=passed,
                       dataset_sha256=digest(directory/'decisions.jsonl'), **pair)
            (directory/'gate.json').write_text(json.dumps(row, indent=2), encoding='utf-8')
            if not passed:
                raise RuntimeError('Action/observation parity failed; dataset NOT approved for training')
            from scripts.audit_route_shadow import audit
            audit_result = audit(directory)
            if not audit_result['all_pop_events_complete']:
                raise RuntimeError('New collection requires a complete v2 pop-event stream')
            (directory/'audit.json').write_text(json.dumps(audit_result, indent=2), encoding='utf-8')
            completed.append(row)
            (output/'progress.json').write_text(json.dumps(dict(completed=completed,
                completed_pairs=len(completed), planned_pairs=len(seeds)), indent=2), encoding='utf-8')
            print(f"PARITY PASS seed={seed} score={pair['shadow']['score']}", flush=True)
        (output/'summary.json').write_text(json.dumps(dict(completed=completed, all_passed=True),
                                                       indent=2), encoding='utf-8')
    except Exception as exc:
        (output/'failed.json').write_text(json.dumps(dict(error=str(exc), completed=len(completed))),
                                          encoding='utf-8')
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seeds', type=int, nargs='+', required=True)
    parser.add_argument('--flight', choices=('baseline', 'shadow'), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.flight:
        if len(args.seeds) != 1:
            parser.error('One seed per child flight')
        flight(args.checkpoint, args.output, args.seeds[0], args.flight == 'shadow')
    else:
        run(args.checkpoint, args.output, args.seeds)
