"""CPU-only matched official flights of frozen PPO and bounded preview MPC+PPO.

The first pair is a zero-gain action/observation parity gate. Subsequent
predeclared development seeds compare active MPC to the identical frozen PPO.
This runner never changes the simulator or passes seed/truth into the agent.
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

import numpy as np

from scripts.collect_route_shadow import digest, update_hash, paired_gate

ROOT = Path(__file__).resolve().parents[1]
ANCHOR_SEED = 261006001
DEVELOPMENT_SEEDS = (261006020, 261006021, 261006022)


def flight(checkpoint, output, seed, label):
    from BalloonPoppingGymEnv.envs.balloon_world import BalloonPoppingEnv
    from BalloonPoppingGymEnv.evaluation.evaluate import load_scenario_parameters
    from BalloonPoppingGymEnv.agents.multiscenario_ppo_policy_agent import MultiScenarioPPOPolicyAgent
    from BalloonPoppingGymEnv.agents.preview_mpc_ppo_agent import PreviewMPCPPOAgent

    parameters, given = load_scenario_parameters(4)
    parameters = copy.deepcopy(parameters)
    parameters['scenario']['random_seed'] = seed
    agent = (MultiScenarioPPOPolicyAgent(given, checkpoint=str(checkpoint))
             if label == 'parent' else PreviewMPCPPOAgent(
                 given, checkpoint=str(checkpoint), mpc_gain=0. if label == 'bypass' else 1.))
    env = BalloonPoppingEnv(render_mode=None, parameters=parameters)
    actions, observations = hashlib.sha256(), hashlib.sha256()
    start = time.monotonic()
    try:
        obs, info = env.reset(seed=seed)
        update_hash(observations, obs)
        status = np.asarray(obs['balloon_status']).reshape(-1).copy()
        events = []
        terminated = truncated = False
        steps = 0
        while not (terminated or truncated):
            action = agent.get_action(obs)
            update_hash(actions, action)
            obs, _, terminated, truncated, info = env.step(action)
            update_hash(observations, obs)
            observed = np.asarray(obs['balloon_status']).reshape(-1)
            for index in np.flatnonzero((observed == 2) & (status != 2)):
                events.append([int(index), float(obs['simulation_time'])])
            status = observed.copy()
            steps += 1
        score = int(info['popped_count'])
        if score != len(events):
            raise RuntimeError('Observed pop transitions and official score differ')
        result = dict(seed=seed, label=label, score=score, steps=steps,
                      final_time=float(obs['simulation_time']), terminated=bool(terminated),
                      truncated=bool(truncated), action_sha256=actions.hexdigest(),
                      observation_sha256=observations.hexdigest(), pop_events=events,
                      mpc_diagnostics=getattr(agent, 'mpc_diagnostics', None),
                      wall_seconds=time.monotonic()-start)
        (output/f'{label}.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
        print(f'COMPLETE seed={seed} label={label} score={score} steps={steps}', flush=True)
    finally:
        env.close()


def run(checkpoint, output):
    checkpoint, output = Path(checkpoint).resolve(), Path(output).resolve()
    manifest_path = checkpoint.parent/'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    bundle = ROOT/'bundle_manifest.json'
    frozen = {ROOT/name: sha for name, sha in manifest['source_sha256'].items()}
    if bundle.exists():
        for name, sha in json.loads(bundle.read_text())['files'].items():
            path = ROOT/name
            if path in frozen and frozen[path] != sha:
                raise ValueError(f'Conflicting frozen source manifests: {name}')
            frozen[path] = sha
    for path, sha in frozen.items():
        if digest(path) != sha:
            raise ValueError(f'Frozen source differs: {path}')
    for name in ('scripts/compare_preview_mpc.py', 'scripts/collect_route_shadow.py',
                 'BalloonPoppingGymEnv/agents/multiscenario_ppo_policy_agent.py',
                 'BalloonPoppingGymEnv/agents/preview_mpc.py',
                 'BalloonPoppingGymEnv/agents/preview_mpc_ppo_agent.py'):
        frozen[ROOT/name] = digest(ROOT/name)
    frozen[checkpoint] = digest(checkpoint)
    frozen[manifest_path] = digest(manifest_path)
    def check_frozen():
        if any(digest(path) != sha for path, sha in frozen.items()):
            raise RuntimeError('Source or checkpoint changed during MPC comparison')
    output.mkdir(parents=True, exist_ok=False)
    plan = dict(scenario=4, anchor_seed=ANCHOR_SEED,
                development_seeds=list(DEVELOPMENT_SEEDS),
                labels=['parent', 'bypass', 'mpc'],
                checkpoint_sha256=digest(checkpoint),
                frozen_sha256={str(p): sha for p, sha in frozen.items()},
                device='cpu', workers=1, timeout_per_flight_seconds=1500,
                evaluation='fresh official Scenario 4; one training-pool bypass seed '
                           'and three unused development seeds', uploaded=False)
    (output/'plan.json').write_text(json.dumps(plan, indent=2), encoding='utf-8')
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1',
                       OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
                       PYTHONPATH=os.pathsep.join([str(ROOT/'ActiveRocketPy'), str(ROOT)]))
    completed = []
    try:
        for seed in (ANCHOR_SEED,) + DEVELOPMENT_SEEDS:
            directory = output/str(seed)
            directory.mkdir()
            labels = ('parent', 'bypass') if seed == ANCHOR_SEED else ('parent', 'mpc')
            pair = {}
            for label in labels:
                check_frozen()
                command = [sys.executable, '-m', 'scripts.compare_preview_mpc',
                           '--checkpoint', str(checkpoint), '--output', str(directory),
                           '--seed', str(seed), '--flight', label]
                with (directory/f'{label}.log').open('x', encoding='utf-8') as log:
                    subprocess.run(command, cwd=ROOT, env=environment, stdout=log,
                                   stderr=subprocess.STDOUT, check=True, timeout=1500)
                check_frozen()
                pair[label] = json.loads((directory/f'{label}.json').read_text())
            if seed == ANCHOR_SEED and not paired_gate(pair['parent'], pair['bypass']):
                raise RuntimeError('Zero MPC gain did not reproduce the parent exactly')
            if any(not row['terminated'] or row['truncated'] for row in pair.values()):
                raise RuntimeError('Incomplete official flight')
            record = dict(seed=seed, **pair)
            completed.append(record)
            (output/'progress.json').write_text(json.dumps(
                dict(completed=completed, planned_pairs=4), indent=2), encoding='utf-8')
            print(f"PAIR COMPLETE seed={seed} scores={[(k, v['score']) for k, v in pair.items()]}",
                  flush=True)
        (output/'summary.json').write_text(json.dumps(dict(
            completed=completed, all_complete=True,
            parent_scores=[r['parent']['score'] for r in completed[1:]],
            mpc_scores=[r['mpc']['score'] for r in completed[1:]]), indent=2), encoding='utf-8')
    except Exception as exc:
        (output/'failed.json').write_text(json.dumps(dict(error=str(exc),
                completed_pairs=len(completed)), indent=2), encoding='utf-8')
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, help=argparse.SUPPRESS)
    parser.add_argument('--flight', choices=('parent', 'bypass', 'mpc'), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.flight:
        if args.seed is None:
            parser.error('Child flight requires seed')
        flight(args.checkpoint, args.output, args.seed, args.flight)
    else:
        run(args.checkpoint, args.output)
