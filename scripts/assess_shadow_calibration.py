"""Offline, field-separated calibration of shadow closest-distance errors.

Future diagnostic outcomes are labels, never runtime features. This calibrates
a distance bias/envelope; it does NOT learn improved 6-DOF dynamics or certify
collision probabilities. Reports with changed references are excluded.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def rows(report, *, fixed_only=True):
    if (report.get('scenario') != 4 or not report.get('terminated') or
            report.get('truncated') or not report.get('scope', '').startswith(
                'Evaluator-only fresh Scenario 4 shadow validation')):
        raise ValueError('Complete official Scenario-4 shadow report required')
    result = []
    for anchor in report['anchors']:
        for prediction in anchor.get('predictions', []):
            if 'actual_closest_distance' not in prediction:
                continue
            if fixed_only and (prediction.get('plan_changes') != 0 or
                               prediction.get('target_changes') != 0):
                continue
            horizon = float(prediction['horizon'])
            predicted = float(prediction['predicted_closest_distance'])
            actual = float(prediction['actual_closest_distance'])
            if not np.all(np.isfinite([horizon, predicted, actual])):
                raise ValueError('Nonfinite diagnostic distance')
            if not 0 < horizon <= 1.0 or min(predicted, actual) < 0:
                continue
            result.append(dict(seed=int(report['seed']),
                               target=int(anchor['target_index']),
                               horizon=horizon, predicted=predicted,
                               actual=actual))
    return result


def fit(training):
    """Equal-weight targets/fields, not thousands of correlated time rows."""
    fitted = {}
    for horizon in sorted({r['horizon'] for r in training}):
        group = [r for r in training if r['horizon'] == horizon]
        clusters = sorted({(r['seed'], r['target']) for r in group})
        residuals = [r['actual'] - r['predicted'] for r in group]
        cluster_bias = [np.median([r['actual'] - r['predicted'] for r in group
                                  if (r['seed'], r['target']) == key])
                        for key in clusters]
        bias = float(np.median(cluster_bias))
        # This is an empirical pointwise envelope, NOT a formal guarantee.
        upper = max(bias, float(np.quantile(residuals, 0.9)))
        fitted[str(horizon)] = dict(bias_m=bias, upper_residual_m=upper,
                                   training_rows=len(group),
                                   training_target_fields=len(clusters))
    return fitted


def metrics(group, bias, upper, radius):
    predicted = np.array([r['predicted'] for r in group])
    actual = np.array([r['actual'] for r in group])
    corrected = np.maximum(0., predicted + bias)
    robust = predicted + upper <= radius
    nominal = predicted <= radius
    shifted = corrected <= radius
    positives = {(r['seed'], r['target']) for r in group if r['actual'] <= radius}
    negatives = {(r['seed'], r['target']) for r in group if r['actual'] > radius}
    return dict(rows=len(group),
                raw_mae_m=float(np.mean(np.abs(predicted-actual))),
                corrected_mae_m=float(np.mean(np.abs(corrected-actual))),
                raw_false_capture_count=int(np.sum(nominal & (actual > radius))),
                raw_capture_predictions=int(np.sum(nominal)),
                corrected_false_capture_count=int(np.sum(shifted & (actual > radius))),
                corrected_capture_predictions=int(np.sum(shifted)),
                upper_bound_false_capture_count=int(np.sum(robust & (actual > radius))),
                upper_bound_capture_predictions=int(np.sum(robust)),
                upper_envelope_exceedance_fraction=float(np.mean(actual > predicted+upper)),
                actual_hit_target_fields=len(positives),
                actual_miss_target_fields=len(negatives))


def assess(train_reports, test_reports, radius=1.5):
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError('Positive finite capture radius required')
    train_seeds = {int(r['seed']) for r in train_reports}
    test_seeds = {int(r['seed']) for r in test_reports}
    if not train_seeds or not test_seeds or train_seeds & test_seeds:
        raise ValueError('Training and testing fields must be nonempty and disjoint')
    training = [row for report in train_reports for row in rows(report)]
    testing = [row for report in test_reports for row in rows(report)]
    calibration = fit(training)
    evaluated = {}
    all_testing = [row for report in test_reports for row in rows(report, fixed_only=False)]
    coverage = {}
    for horizon in sorted({r['horizon'] for r in testing}):
        key = str(horizon)
        if key not in calibration:
            continue
        model = calibration[key]
        evaluated[key] = metrics([r for r in testing if r['horizon'] == horizon],
                                 model['bias_m'], model['upper_residual_m'], radius)
        all_group = [r for r in all_testing if r['horizon'] == horizon]
        fixed_group = [r for r in testing if r['horizon'] == horizon]
        coverage[key] = dict(all_completed_rows=len(all_group),
                            fixed_plan_rows=len(fixed_group),
                            all_actual_capture_rows=sum(r['actual'] <= radius for r in all_group),
                            fixed_plan_actual_capture_rows=sum(r['actual'] <= radius for r in fixed_group))
    # Predeclared conservative screen. A pass would justify another experiment,
    # not installation in the qualification runner. No test-fitted parameters.
    half = evaluated.get('0.5', {})
    passed = bool(half and len(train_seeds) >= 2 and
                  half['rows'] >= 20 and half['actual_hit_target_fields'] >= 3 and
                  half['actual_miss_target_fields'] >= 3 and
                  half['corrected_mae_m'] <= 0.9*half['raw_mae_m'] and
                  half['upper_envelope_exceedance_fraction'] <= .1 and
                  half['upper_bound_capture_predictions'] >= 5 and
                  half['upper_bound_false_capture_count'] == 0)
    return dict(schema='shadow_distance_calibration_diagnostic_v1',
                training_seeds=sorted(train_seeds), test_seeds=sorted(test_seeds),
                calibration=calibration, test_metrics=evaluated,
                test_coverage=coverage,
                capture_radius_m=radius, controller_experiment_gate_passed=passed,
                live_agent_changed=False,
                limitations=('Correlated anchors; fixed-plan filtering can exclude successful '
                             'captures because a pop/replan changes the reference. Check coverage. '
                             'Empirical distances, not certified capture probabilities.'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train', type=Path, nargs='+', required=True)
    parser.add_argument('--test', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    reports = {path: json.loads(path.read_text(encoding='utf-8'))
               for path in (*args.train, *args.test)}
    result = assess([reports[p] for p in args.train], [reports[p] for p in args.test])
    result['input_sha256'] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in reports}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
