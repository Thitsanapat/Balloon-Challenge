"""Pool two audited selected-agent route archives and gate frozen weights.

This offline script never imports a flight environment or writes a live agent.
It leaves a diagnostic report even when calibration fails, and creates a
weights-only artifact only after the predeclared group-held-out gate passes.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from BalloonPoppingGymEnv.agents.scenario4_learned_route_agent import (
    FEATURE_NAMES, FrozenCaptureModel,
)
from scripts.diagnose_route_capture_ml import (
    fit_regularized_logistic, held_out_report, load_audited_attempts,
)
from scripts.package_guarded_capture_model import TRAINING_SEEDS


SECOND_SEEDS = (
    10298371, 68394312, 50192733, 482617990,
    297410583, 700126354, 846320117, 119657428,
    337854091, 914708236, 592461073, 781235409,
)
MIN_BRIER_GAIN = 0.003
MIN_LOG_LOSS_GAIN = 0.006


def calibration_gate(report):
    if (report.get('fields') != 24 or report.get('attempts', 0) < 100 or
            tuple(report.get('feature_names', ())) != FEATURE_NAMES):
        return False
    prior, learned = report['prior'], report['learned']
    values = [prior['brier'], learned['brier'], prior['log_loss'],
              learned['log_loss']]
    return bool(np.all(np.isfinite(values)) and
                prior['brier'] - learned['brier'] >= MIN_BRIER_GAIN and
                prior['log_loss'] - learned['log_loss'] >= MIN_LOG_LOSS_GAIN)


def verified_records(archive, expected_seeds):
    archive = Path(archive)
    records = load_audited_attempts(archive)
    observed_seeds = {row['seed'] for row in records}
    if observed_seeds != set(expected_seeds):
        raise ValueError(f'Audited field set differs from declared seeds: {archive}')
    return records, hashlib.sha256(archive.read_bytes()).hexdigest()


def fit(first_archive, second_archive, report_path, weights_path):
    first_archive, second_archive = Path(first_archive), Path(second_archive)
    report_path, weights_path = Path(report_path), Path(weights_path)
    if (first_archive.resolve() == second_archive.resolve() or
            report_path.resolve() == weights_path.resolve() or
            report_path.exists() or weights_path.exists()):
        raise ValueError('Distinct new archive/report/weights paths required')
    first, first_sha = verified_records(first_archive, TRAINING_SEEDS)
    second, second_sha = verified_records(second_archive, SECOND_SEEDS)
    records = first + second
    if len({row['seed'] for row in records}) != 24:
        raise ValueError('Training archives overlap')
    report = held_out_report(records)
    full_model = fit_regularized_logistic(
        np.asarray([row['features'] for row in records]),
        np.asarray([row['hit'] for row in records]))
    report['full_fit_prototype'] = dict(
        center=full_model[0].tolist(), scale=full_model[1].tolist(),
        coefficients=full_model[2].tolist(),
        role='offline frozen candidate only; never auto-promoted')
    report['training_archive_sha256'] = [first_sha, second_sha]
    report['predeclared_gate'] = dict(min_brier_gain=MIN_BRIER_GAIN,
                                      min_log_loss_gain=MIN_LOG_LOSS_GAIN,
                                      passed=calibration_gate(report))
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    if not report['predeclared_gate']['passed']:
        return report
    provenance = hashlib.sha256((first_sha + second_sha).encode('ascii')).hexdigest()
    payload = dict(schema='guarded_capture_risk_v1',
                   feature_names=list(FEATURE_NAMES),
                   center=full_model[0].tolist(), scale=full_model[1].tolist(),
                   coefficients=full_model[2].tolist(),
                   source_archive_sha256=provenance)
    FrozenCaptureModel(payload)
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    weights_path.write_text(json.dumps(payload, indent=2) + '\n',
                            encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('first_archive', type=Path)
    parser.add_argument('second_archive', type=Path)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--weights', type=Path, required=True)
    args = parser.parse_args()
    report = fit(args.first_archive, args.second_archive,
                 args.report, args.weights)
    print(json.dumps(dict(fields=report['fields'], attempts=report['attempts'],
                          prior=report['prior'], learned=report['learned'],
                          passed=report['predeclared_gate']['passed'])))


if __name__ == '__main__':
    main()
