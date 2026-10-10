"""Run a frozen second paired holdout without changing the live policy.

All eight seeds were checked against used_seeds.json before they were added
to used_seed_manual.jsonl. The first four holdout outcomes were already known
when this new, independent set was declared. Simulations remain serial and
CPU-only through validate_guarded_route_ml.run().
"""

from pathlib import Path

from scripts import validate_guarded_route_ml as validation


SEEDS = (
    140200341, 1098270439, 1904235602, 568300771,
    178420909, 820431975, 1126043508, 662504311,
)
RESULT = (Path(__file__).resolve().parents[1] /
          'BalloonPoppingGymEnv/evaluation/results/guarded_route_ml_stage2_20261008.json')


def run():
    validation.SEEDS = SEEDS
    validation.RESULT = RESULT
    validation.run()


if __name__ == '__main__':
    run()
