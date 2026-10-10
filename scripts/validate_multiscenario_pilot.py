"""Small, CPU-only paired validation of a frozen mixed-scenario checkpoint.

Uses the unmodified official environment, new validation seeds, and the exact
zero-residual training controller as baseline. No training or upload occurs.
Run each flight in a fresh subprocess so imported-source fingerprints match.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import yaml
from scripts.summarize_paired_evaluations import summarize

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEEDS = (1000000007, 1111111111, 1428571429)
# Predeclared with random.Random(261005).sample(range(1500000000,2000000000),10).
# This draws seed identifiers only; no field/score was inspected to choose them.
HOLDOUT_SEEDS = (1793692987, 1576325320, 1931191098, 1735313215, 1889085391,
                 1877745918, 1818045955, 1968662391, 1726021446, 1717629685)
SEED_POOLS = {'validation': (1000000000, 1500000000),
              'holdout': (1500000000, 2000000000)}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_seeds(seeds, training_rows, split='validation', prior_seeds=()):
    if split not in SEED_POOLS:
        raise ValueError('Unknown evaluation split')
    low, high = SEED_POOLS[split]
    if (not seeds or len(set(seeds)) != len(seeds)
            or any(type(s) is not int or not low <= s < high for s in seeds)):
        raise ValueError(f'Unique seeds from the reserved {split} pool required')
    if set(seeds) & {row['seed'] for row in training_rows}:
        raise ValueError('Validation seed was used in training')
    if set(seeds) & set(prior_seeds):
        raise ValueError('Evaluation seed was already used in a prior plan')


def configurations(checkpoint, baseline_checkpoint=None):
    common = dict(scenario_number=4, render_mode=None, leaderboard_submission=False)
    result = {
        'baseline': dict(common, agent_name='Frozen zero-residual full-route controller',
                         agent_module_path='./BalloonPoppingGymEnv/agents/multiscenario_residual_agent.py',
                         agent_class_name='MultiScenarioResidualController', agent_kwargs={}),
        'candidate': dict(common, agent_name='Frozen mixed-scenario PPO deterministic validation',
                          agent_module_path='./BalloonPoppingGymEnv/agents/multiscenario_ppo_policy_agent.py',
                          agent_class_name='MultiScenarioPPOPolicyAgent',
                          agent_kwargs=dict(checkpoint=str(Path(checkpoint).resolve()))),
    }
    if baseline_checkpoint is not None:
        result['baseline'] = dict(common, agent_name='Frozen learned parent PPO',
            agent_module_path='./BalloonPoppingGymEnv/agents/multiscenario_ppo_policy_agent.py',
            agent_class_name='MultiScenarioPPOPolicyAgent',
            agent_kwargs=dict(checkpoint=str(Path(baseline_checkpoint).resolve())))
    return result


def run(checkpoint, output, seeds, split='validation', prior_plans=(), baseline_checkpoint=None):
    checkpoint, output = Path(checkpoint).resolve(), Path(output).resolve()
    manifest_path = checkpoint.parent/'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    training_rows = [json.loads(s) for s in (checkpoint.parent/'episode_seeds.jsonl').read_text().splitlines() if s.strip()]
    baseline_checkpoint = Path(baseline_checkpoint).resolve() if baseline_checkpoint is not None else None
    baseline_manifest = None
    if baseline_checkpoint is not None:
        if baseline_checkpoint == checkpoint or digest(baseline_checkpoint) == digest(checkpoint):
            raise ValueError('Learned baseline and candidate must be distinct checkpoints')
        baseline_manifest = json.loads((baseline_checkpoint.parent/'manifest.json').read_text())
        training_rows += [json.loads(s) for s in (baseline_checkpoint.parent/'episode_seeds.jsonl').read_text().splitlines() if s.strip()]
    prior_plans = [Path(p).resolve() for p in prior_plans]
    if split == 'holdout' and not prior_plans:
        raise ValueError('Holdout requires prior evaluation plans for overlap checks')
    prior_seeds = [s for p in prior_plans for s in json.loads(p.read_text())['seeds']]
    validate_seeds(seeds, training_rows, split, prior_seeds)
    expected = manifest['source_sha256']
    for name, sha in expected.items():
        if digest(ROOT/name) != sha:
            raise ValueError(f'Training source mismatch: {name}')
    # Runtime snapshot also covers RocketPy and weather input, not just agents.
    bundle = ROOT/'bundle_manifest.json'
    if bundle.exists():
        for name, sha in json.loads(bundle.read_text())['files'].items():
            if digest(ROOT/name) != sha:
                raise ValueError(f'Runtime snapshot mismatch: {name}')
    paths = {ROOT/name for name in expected}
    if baseline_manifest is not None:
        for name, sha in baseline_manifest['source_sha256'].items():
            if digest(ROOT/name) != sha:
                raise ValueError(f'Baseline training source mismatch: {name}')
            paths.add(ROOT/name)
        paths.update((baseline_checkpoint, baseline_checkpoint.parent/'manifest.json',
                      baseline_checkpoint.parent/'episode_seeds.jsonl'))
    if bundle.exists():
        paths.update(ROOT/name for name in json.loads(bundle.read_text())['files'])
    paths.update((checkpoint, manifest_path, checkpoint.parent/'episode_seeds.jsonl', Path(__file__).resolve(),
                  ROOT/'scripts/evaluate_guidance.py', ROOT/'scripts/summarize_paired_evaluations.py',
                  ROOT/'BalloonPoppingGymEnv/agents/multiscenario_ppo_policy_agent.py'))
    paths.update(prior_plans)
    frozen = {str(p): digest(p) for p in sorted(paths)}
    output.mkdir(parents=True, exist_ok=False)
    def save(name, data):
        (output/name).write_text(json.dumps(data, indent=2, allow_nan=False), encoding='utf-8')
    def check_frozen():
        if any(digest(p) != sha for p, sha in frozen.items()):
            raise RuntimeError('Frozen source/model changed during validation')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1',
               OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    configs = configurations(checkpoint, baseline_checkpoint)
    for label, config in configs.items():
        (output/f'{label}.yaml').write_text(yaml.safe_dump(config), encoding='utf-8')
    save('plan.json', dict(scenario=4, seeds=list(seeds), split=split,
                          prior_plans=[str(p) for p in prior_plans],
                          checkpoint_sha256=digest(checkpoint), frozen_sha256=frozen,
                          baseline_checkpoint_sha256=digest(baseline_checkpoint) if baseline_checkpoint else None,
                          configurations=configs, device='cpu', workers=1,
                          timeout_per_flight_seconds=1200, uploaded=False))
    started = time.monotonic()
    completed = []
    reports = {label: [] for label in configs}
    try:
        for seed in seeds:
            for label in configs:
                check_frozen()
                save('status.json', dict(status='running', current=dict(seed=seed, agent=label),
                                         completed=completed, wall_seconds=time.monotonic()-started))
                report = output/f'{label}_{seed}.json'
                with (output/f'{label}_{seed}.log').open('w', encoding='utf-8') as log:
                    subprocess.run([sys.executable, '-u', '-m', 'scripts.evaluate_guidance',
                                    str(output/f'{label}.yaml'), '--seeds', str(seed),
                                    '--report', str(report)], cwd=ROOT, env=env,
                                   stdout=log, stderr=subprocess.STDOUT, check=True, timeout=1200)
                check_frozen()
                row = json.loads(report.read_text())[0]
                reports[label].append(report)
                completed.append(dict(seed=seed, agent=label, score=row['score'],
                                      terminated=row['terminated'], truncated=row['truncated']))
                print(json.dumps(completed[-1]), flush=True)
                if row['truncated'] or not row['terminated']:
                    raise RuntimeError('Incomplete official flight; do not silently replace seed')
        summary = summarize(reports['baseline'], reports['candidate'])
        check_frozen()
        summary['checkpoint_sha256'] = digest(checkpoint)
        summary['baseline_checkpoint_sha256'] = digest(baseline_checkpoint) if baseline_checkpoint else None
        summary['frozen_sources_and_model_unchanged'] = True
        summary['split'] = split
        summary['promotion'] = ('Not automatic: review paired means, low scores and all failures; '
                                'this set becomes development data if used to tune the next model')
        save('summary.json', summary)
        save('status.json', dict(status='completed', completed=completed,
                                wall_seconds=time.monotonic()-started, uploaded=False))
    except BaseException as error:
        save('status.json', dict(status='failed', error=repr(error), completed=completed,
                                wall_seconds=time.monotonic()-started, uploaded=False))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--baseline-checkpoint', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--split', choices=tuple(SEED_POOLS), default='validation')
    parser.add_argument('--prior-plan', type=Path, action='append', default=[])
    parser.add_argument('--seeds', type=int, nargs='+')
    args = parser.parse_args()
    seeds = args.seeds if args.seeds is not None else (HOLDOUT_SEEDS if args.split == 'holdout' else DEFAULT_SEEDS)
    run(args.checkpoint, args.output, seeds, args.split, args.prior_plan, args.baseline_checkpoint)
