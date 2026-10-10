"""Five predeclared, unseen Scenario-4 pairs for a promising MPC+PPO pilot.

Run only after inspecting the development pilot. Every comparison uses fresh
official flights and the same frozen checkpoint; the only agent difference is
the bounded online preview correction. No tuning uses these validation seeds.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from scripts.collect_route_shadow import digest

ROOT = Path(__file__).resolve().parents[1]
# random.Random(261007).sample(range(1_000_000_000, 1_500_000_000), 5),
# declared before inspecting any of these fields or scores.
SEEDS = (1141149390, 1165092567, 1410104461, 1399585391, 1352273614)


def validate_seed_plan(prior_plans):
    if any(not 1_000_000_000 <= seed < 1_500_000_000 for seed in SEEDS):
        raise ValueError('Validation seeds must be in the reserved pool')
    prior = set()
    for path in prior_plans:
        data = json.loads(Path(path).read_text())
        prior.update(data.get('seeds', []))
        prior.update(data.get('development_seeds', []))
        if data.get('anchor_seed') is not None:
            prior.add(data['anchor_seed'])
    if len(SEEDS) != len(set(SEEDS)) or set(SEEDS) & prior:
        raise ValueError('Validation seeds overlap a previous evaluation plan')


def run(checkpoint, output, pilot_plan, prior_plans):
    checkpoint, output = Path(checkpoint).resolve(), Path(output).resolve()
    pilot_plan = Path(pilot_plan).resolve()
    prior_plans = [Path(p).resolve() for p in prior_plans]
    if not (pilot_plan.parent/'summary.json').exists():
        raise ValueError('Development pilot must complete before validation')
    validate_seed_plan([pilot_plan] + prior_plans)
    pilot = json.loads(pilot_plan.read_text())
    if pilot['checkpoint_sha256'] != digest(checkpoint):
        raise ValueError('Checkpoint differs from development pilot')
    frozen = {Path(path): sha for path, sha in pilot['frozen_sha256'].items()}
    frozen[Path(__file__).resolve()] = digest(__file__)
    frozen[pilot_plan] = digest(pilot_plan)
    for path in prior_plans:
        frozen[path] = digest(path)
    def check_frozen():
        if any(digest(path) != sha for path, sha in frozen.items()):
            raise RuntimeError('Frozen source, model, or prior plan changed')
    check_frozen()
    output.mkdir(parents=True, exist_ok=False)
    plan = dict(scenario=4, seeds=list(SEEDS), split='previously_unseen_validation',
                prior_plans=[str(p) for p in [pilot_plan] + prior_plans],
                checkpoint_sha256=digest(checkpoint),
                frozen_sha256={str(p): sha for p, sha in frozen.items()},
                device='cpu', workers=1, timeout_per_flight_seconds=1500,
                source='official simulator; frozen parent and MPC+PPO', uploaded=False)
    (output/'plan.json').write_text(json.dumps(plan, indent=2), encoding='utf-8')
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1',
                       OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
                       PYTHONPATH=os.pathsep.join([str(ROOT/'ActiveRocketPy'), str(ROOT)]))
    completed = []
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
                pair[label] = json.loads((directory/f'{label}.json').read_text())
                if not pair[label]['terminated'] or pair[label]['truncated']:
                    raise RuntimeError(f'Incomplete official flight seed={seed} label={label}')
            completed.append(dict(seed=seed, **pair))
            (output/'progress.json').write_text(json.dumps(dict(
                completed=completed, planned_pairs=len(SEEDS)), indent=2), encoding='utf-8')
            print(f"PAIR COMPLETE seed={seed} parent={pair['parent']['score']} "
                  f"mpc={pair['mpc']['score']}", flush=True)
        (output/'summary.json').write_text(json.dumps(dict(
            completed=completed, all_complete=True,
            parent_scores=[r['parent']['score'] for r in completed],
            mpc_scores=[r['mpc']['score'] for r in completed]), indent=2), encoding='utf-8')
    except Exception as exc:
        (output/'failed.json').write_text(json.dumps(dict(error=str(exc),
                completed_pairs=len(completed)), indent=2), encoding='utf-8')
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--pilot-plan', type=Path, required=True)
    parser.add_argument('--prior-plan', type=Path, action='append', default=[])
    args = parser.parse_args()
    run(args.checkpoint, args.output, args.pilot_plan, args.prior_plan)
