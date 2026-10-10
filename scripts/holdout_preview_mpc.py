"""Ten fresh Scenario-4 pairs after the frozen MPC+PPO validation screen.

This is an assessment, not a tuning loop. The one CPU worker runs each agent
in a separate official simulator process. Seed IDs stay outside both agents.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from scripts.collect_route_shadow import digest

ROOT = Path(__file__).resolve().parents[1]
# random.Random(261008).sample(range(1_500_000_000, 2_000_000_000), 10),
# fixed before inspecting any of these fields or their scores.
SEEDS = (1638895464, 1584252769, 1722428225, 1859368188, 1570091972,
         1824032641, 1945086148, 1533739991, 1676901674, 1830415931)


def validate_seed_plan(prior_plans):
    if any(not 1_500_000_000 <= seed < 2_000_000_000 for seed in SEEDS):
        raise ValueError('Holdout seeds must be in the reserved pool')
    prior = set()
    for path in prior_plans:
        data = json.loads(Path(path).read_text(encoding='utf-8'))
        prior.update(data.get('seeds', []))
        prior.update(data.get('development_seeds', []))
        if data.get('anchor_seed') is not None:
            prior.add(data['anchor_seed'])
    if len(SEEDS) != len(set(SEEDS)) or set(SEEDS) & prior:
        raise ValueError('Holdout seeds overlap a previous evaluation plan')


def run(checkpoint, output, validation_plan, prior_plans):
    checkpoint, output = Path(checkpoint).resolve(), Path(output).resolve()
    validation_plan = Path(validation_plan).resolve()
    prior_plans = [Path(p).resolve() for p in prior_plans]
    if not (validation_plan.parent/'summary.json').exists():
        raise ValueError('Five-seed validation must complete before holdout')
    validate_seed_plan([validation_plan] + prior_plans)
    validation = json.loads(validation_plan.read_text(encoding='utf-8'))
    completed = json.loads((validation_plan.parent/'summary.json').read_text(encoding='utf-8'))
    rows = completed['completed']
    if (not completed['all_complete'] or [r['seed'] for r in rows] != validation['seeds']):
        raise ValueError('Incomplete validation pairs')
    deltas = [r['mpc']['score']-r['parent']['score'] for r in rows]
    if sum(deltas) <= 0 or sum(d > 0 for d in deltas) < 3 or min(deltas) < -2:
        raise ValueError('Predeclared validation screen did not pass')
    if validation['checkpoint_sha256'] != digest(checkpoint):
        raise ValueError('Checkpoint differs from five-seed validation')
    frozen = {Path(path): sha for path, sha in validation['frozen_sha256'].items()}
    frozen[Path(__file__).resolve()] = digest(__file__)
    frozen[validation_plan] = digest(validation_plan)
    frozen[validation_plan.parent/'summary.json'] = digest(validation_plan.parent/'summary.json')
    for path in prior_plans:
        frozen[path] = digest(path)

    def check_frozen():
        if any(digest(path) != sha for path, sha in frozen.items()):
            raise RuntimeError('Frozen source, model, or prior plan changed')

    check_frozen()
    output.mkdir(parents=True, exist_ok=False)
    plan = dict(scenario=4, seeds=list(SEEDS), split='previously_unseen_holdout',
                previous_validation=str(validation_plan),
                prior_plans=[str(p) for p in prior_plans],
                previous_validation_differences=deltas,
                checkpoint_sha256=digest(checkpoint),
                frozen_sha256={str(p): sha for p, sha in frozen.items()},
                device='cpu', workers=1, timeout_per_flight_seconds=1500,
                source='official simulator; frozen parent and MPC+PPO', uploaded=False)
    (output/'plan.json').write_text(json.dumps(plan, indent=2), encoding='utf-8')
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1',
                       OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
                       PYTHONPATH=os.pathsep.join([str(ROOT/'ActiveRocketPy'), str(ROOT)]))
    results = []
    try:
        for seed in SEEDS:
            directory = output/str(seed)
            directory.mkdir()
            pair = {}
            for label in ('parent', 'mpc'):
                check_frozen()
                command = [sys.executable, '-m', 'scripts.compare_preview_mpc',
                           '--checkpoint', str(checkpoint), '--output', str(directory),
                           '--seed', str(seed), '--flight', label]
                with (directory/f'{label}.log').open('x', encoding='utf-8') as log:
                    subprocess.run(command, cwd=ROOT, env=environment, stdout=log,
                                   stderr=subprocess.STDOUT, check=True, timeout=1500)
                check_frozen()
                pair[label] = json.loads((directory/f'{label}.json').read_text(encoding='utf-8'))
                if not pair[label]['terminated'] or pair[label]['truncated']:
                    raise RuntimeError(f'Incomplete official flight seed={seed} label={label}')
            results.append(dict(seed=seed, **pair))
            (output/'progress.json').write_text(json.dumps(dict(
                completed=results, planned_pairs=len(SEEDS)), indent=2), encoding='utf-8')
            print(f"PAIR COMPLETE seed={seed} parent={pair['parent']['score']} "
                  f"mpc={pair['mpc']['score']}", flush=True)
        (output/'summary.json').write_text(json.dumps(dict(
            completed=results, all_complete=True,
            parent_scores=[r['parent']['score'] for r in results],
            mpc_scores=[r['mpc']['score'] for r in results]), indent=2), encoding='utf-8')
    except Exception as exc:
        (output/'failed.json').write_text(json.dumps(dict(error=str(exc),
                completed_pairs=len(results)), indent=2), encoding='utf-8')
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--validation-plan', type=Path, required=True)
    parser.add_argument('--prior-plan', type=Path, action='append', default=[])
    args = parser.parse_args()
    run(args.checkpoint, args.output, args.validation_plan, args.prior_plan)
