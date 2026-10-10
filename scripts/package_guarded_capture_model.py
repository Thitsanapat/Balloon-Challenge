"""Publish a weights-only experimental model if held-out calibration passes.

The full report contains seed IDs and outcomes for audit. The runtime artifact
contains only learned constants and a training-archive digest, never seed IDs.
"""

import argparse
import json
from pathlib import Path

from BalloonPoppingGymEnv.agents.scenario4_learned_route_agent import (
    FEATURE_NAMES, FrozenCaptureModel,
)


TRAINING_SEEDS = (
    313734695, 48700076, 470928925, 257808910,
    164593090, 300439883, 93120163, 767678882,
    230652457, 333239953, 839958652, 724686099,
)


def package(report_path, output_path):
    report = json.loads(Path(report_path).read_text(encoding='utf-8'))
    if (report.get('schema') != 'route_capture_ml_group_holdout_v1' or
            report.get('fields') != len(TRAINING_SEEDS) or
            {fold['seed'] for fold in report.get('folds', [])} != set(TRAINING_SEEDS) or
            tuple(report.get('feature_names', ())) != FEATURE_NAMES or
            report.get('attempts', 0) < 36 or
            report['learned']['brier'] >= report['prior']['brier'] or
            report['learned']['log_loss'] >= report['prior']['log_loss']):
        raise ValueError('Held-out capture calibration did not pass the gate')
    fitted = report['full_fit_prototype']
    payload = dict(schema='guarded_capture_risk_v1',
                   feature_names=list(FEATURE_NAMES),
                   center=fitted['center'], scale=fitted['scale'],
                   coefficients=fitted['coefficients'],
                   source_archive_sha256=report['source_archive_sha256'])
    FrozenCaptureModel(payload)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + '\n',
                           encoding='utf-8')
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('report', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    package(args.report, args.output)
    print(f'WEIGHTS_ONLY {args.output}')


if __name__ == '__main__':
    main()
