"""Disjoint 12-field validation of one frozen, altitude-aware capture model.

The seven-feature choice was made using only the first audited training pool.
No labels from the second pool are used for fitting or feature selection here.
This is offline calibration, not a route-policy or qualification score test.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from scripts.diagnose_detailed_route_capture import (
    DETAIL_NAMES, load_detailed_audited_attempts,
)
from scripts.diagnose_route_capture_ml import (
    FEATURE_NAMES, fit_regularized_logistic,
    predict_logistic,
)
from scripts.fit_combined_guarded_route import SECOND_SEEDS
from scripts.package_guarded_capture_model import TRAINING_SEEDS


FEATURE_COLUMN = DETAIL_NAMES.index('relative_altitude')
MIN_BRIER_GAIN = 0.003
MIN_LOG_LOSS_GAIN = 0.006


def metric(y, p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    y = np.asarray(y, dtype=float)
    return dict(brier=float(np.mean((p - y) ** 2)),
                log_loss=float(np.mean(-y * np.log(p) -
                                       (1 - y) * np.log1p(-p))))


def validate(first_archive, second_archive):
    first = load_detailed_audited_attempts(first_archive)
    second = load_detailed_audited_attempts(second_archive)
    if ({row['seed'] for row in first} != set(TRAINING_SEEDS) or
            {row['seed'] for row in second} != set(SECOND_SEEDS) or
            set(TRAINING_SEEDS) & set(SECOND_SEEDS)):
        raise ValueError('Unexpected or overlapping audited field groups')
    y_train = np.array([row['hit'] for row in first])
    y_test = np.array([row['hit'] for row in second])
    x_train = np.array([row['features'] for row in first])
    x_test = np.array([row['features'] for row in second])
    columns = list(range(len(FEATURE_NAMES))) + [FEATURE_COLUMN]
    prior = np.full(len(second), (y_train.sum() + 0.5) / (len(first) + 1.0))
    base = predict_logistic(x_test[:, :len(FEATURE_NAMES)],
                            fit_regularized_logistic(
                                x_train[:, :len(FEATURE_NAMES)], y_train))
    detailed = predict_logistic(x_test[:, columns],
                                fit_regularized_logistic(
                                    x_train[:, columns], y_train))
    base_metric = metric(y_test, base)
    detailed_metric = metric(y_test, detailed)
    brier_gain = base_metric['brier'] - detailed_metric['brier']
    log_loss_gain = base_metric['log_loss'] - detailed_metric['log_loss']
    return dict(
        schema='disjoint_altitude_capture_holdout_v1',
        train_fields=len(TRAINING_SEEDS), test_fields=len(SECOND_SEEDS),
        train_attempts=len(first), test_attempts=len(second),
        test_hits=int(y_test.sum()),
        feature_names=list(FEATURE_NAMES) + ['relative_altitude'],
        prior=metric(y_test, prior), baseline_six=base_metric,
        altitude_seven=detailed_metric,
        brier_gain=brier_gain, log_loss_gain=log_loss_gain,
        calibration_gate_passed=bool(
            brier_gain >= MIN_BRIER_GAIN and
            log_loss_gain >= MIN_LOG_LOSS_GAIN),
        agent_connected=False,
        caveat='Calibration on executed choices; no route-score validation',
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('first_archive', type=Path)
    parser.add_argument('second_archive', type=Path)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--wait-for-second', action='store_true')
    args = parser.parse_args()
    if args.report.exists():
        raise ValueError('Refusing to overwrite an existing holdout report')
    if args.wait_for_second:
        deadline = time.monotonic() + 8 * 3600
        second_dir = args.second_archive.with_suffix('')
        while not (args.second_archive.exists() and
                   (second_dir / 'route_capture_fit.json').exists()):
            if (second_dir / 'failed.json').exists():
                raise RuntimeError('Second audited collection failed')
            if time.monotonic() > deadline:
                raise TimeoutError('Second audited collection did not finish')
            time.sleep(30)
        summary = json.loads((second_dir / 'summary.json').read_text())
        if not summary.get('all_passed') or len(summary.get('completed', ())) != 12:
            raise ValueError('Second audited collection did not pass its gate')
    report = validate(args.first_archive, args.second_archive)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
