"""Offline, group-held-out first-leg capture calibration from audited flights.

This script does not run inside an agent.  The logged pre-action observation
features are predictors; later hit/miss outcomes are *labels only*.  Never
train on a held-out seed's decisions and call that seed unseen.  No unflown
route proposal is assigned a made-up outcome.
"""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from zipfile import ZipFile

import numpy as np


FEATURE_NAMES = (
    'log_time_to_deadline', 'relative_range', 'closing_speed',
    'required_lateral_fraction', 'required_total_fraction',
    'available_acceleration',
)


def first_leg_features(decision):
    """Only pre-action, observation-derived quantities from the chosen leg."""
    target = decision['target']
    node = np.asarray(decision['features']['nodes'][target], dtype=float)
    context = np.asarray(decision['features']['context'], dtype=float)
    horizon = float(decision['deadline'] - decision['time'])
    if (node.shape != (12,) or context.shape != (9,) or
            not np.all(np.isfinite(node)) or
            not np.all(np.isfinite(context)) or horizon <= 0 or
            not bool(decision['features']['visible_mask'][target])):
        raise ValueError('Invalid pre-action chosen-leg features')
    relative_position = node[:3] * 150.0
    relative_velocity = node[3:6] * 30.0
    available = max(float(context[8] * 20.0), 1e-6)
    # A constant-acceleration intercept is only a normalized difficulty proxy,
    # not a claim that the real rocket follows this exact motion.
    required = 2.0 * (relative_position + relative_velocity * horizon) / horizon**2
    return np.array((
        np.log1p(horizon), np.linalg.norm(relative_position) / 150.0,
        -float(np.dot(relative_position, relative_velocity)) /
            max(np.linalg.norm(relative_position), 1e-6) / 30.0,
        np.linalg.norm(required[:2]) / available,
        np.linalg.norm(required + np.array([0., 0., 9.80665])) / available,
        available / 20.0,
    ), dtype=float)


def load_audited_attempts(archive_path):
    """Join first chosen decision to its later audited outcome, by attempt ID."""
    records = []
    with ZipFile(archive_path) as archive:
        names = archive.namelist()
        folders = sorted(name[:-len('/audit.json')] for name in names
                         if name.startswith('data/') and
                         name.endswith('/audit.json'))
        if not folders:
            raise ValueError('No audited training fields in archive')
        for folder in folders:
            seed = int(folder.split('/')[-1])
            audit = json.loads(archive.read(folder + '/audit.json'))
            gate = json.loads(archive.read(folder + '/gate.json'))
            if not (audit['parity_verified'] and
                    audit['all_pop_events_complete'] and
                    gate['exact_parity']):
                raise ValueError(f'Unverified flight {seed}')
            raw_stream = archive.read(folder + '/decisions.jsonl')
            if hashlib.sha256(raw_stream).hexdigest() != audit['dataset_sha256']:
                raise ValueError(f'Audited stream digest mismatch on {seed}')
            stream = raw_stream.splitlines()
            first = {}
            outcomes = {}
            for line in stream:
                row = json.loads(line)
                if row['kind'] == 'decision':
                    first.setdefault(int(row['attempt_id']), row)
                elif row['kind'] == 'outcome':
                    attempt_id = int(row['attempt_id'])
                    if attempt_id in outcomes:
                        raise ValueError(f'Duplicate outcome on {seed}')
                    outcomes[attempt_id] = row
            if len(outcomes) != audit['executed_attempts']:
                raise ValueError(f'Incomplete outcomes on {seed}')
            for attempt_id, outcome in sorted(outcomes.items()):
                if outcome['outcome'] not in ('hit', 'miss'):
                    continue  # cancelled/censored attempts are not false hits
                decision = first.get(attempt_id)
                if decision is None or decision['target'] != outcome['target']:
                    raise ValueError(f'Unmatched chosen leg on {seed}')
                records.append(dict(seed=seed, attempt_id=attempt_id,
                                    features=first_leg_features(decision),
                                    hit=int(outcome['outcome'] == 'hit')))
    if len({row['seed'] for row in records}) < 3:
        raise ValueError('At least three independent field groups required')
    return records


def fit_regularized_logistic(x, y, penalty=8.0, iterations=30):
    """Stable small-sample logistic fit, with unpenalized intercept."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if x.ndim != 2 or y.shape != (len(x),) or len(x) < 2:
        raise ValueError('Invalid design matrix')
    center = x.mean(axis=0)
    scale = np.maximum(x.std(axis=0), 0.2)
    design = np.c_[np.ones(len(x)), (x - center) / scale]
    weights = np.zeros(design.shape[1])
    weights[0] = np.log((y.sum() + 0.5) /
                        (len(y) - y.sum() + 0.5))
    ridge = np.diag(np.r_[0.0, np.full(x.shape[1], penalty)])
    for _ in range(iterations):
        logits = np.clip(design @ weights, -20., 20.)
        probabilities = 1.0 / (1.0 + np.exp(-logits))
        curvature = np.maximum(probabilities * (1.0 - probabilities), 1e-5)
        hessian = design.T @ (curvature[:, None] * design) + ridge
        gradient = design.T @ (probabilities - y) + ridge @ weights
        update = np.linalg.solve(hessian, gradient)
        weights -= update
        if np.max(np.abs(update)) < 1e-8:
            break
    return center, scale, weights


def predict_logistic(x, model):
    center, scale, weights = model
    design = np.c_[np.ones(len(x)), (np.asarray(x) - center) / scale]
    return 1.0 / (1.0 + np.exp(-np.clip(design @ weights, -20., 20.)))


def held_out_report(records):
    """Leave each entire simulator field out of fitting and normalization."""
    groups = defaultdict(list)
    for row in records:
        groups[row['seed']].append(row)
    results = []
    for seed in sorted(groups):
        train = [row for key in groups if key != seed for row in groups[key]]
        test = groups[seed]
        x_train = np.array([row['features'] for row in train])
        y_train = np.array([row['hit'] for row in train])
        x_test = np.array([row['features'] for row in test])
        y_test = np.array([row['hit'] for row in test])
        model = fit_regularized_logistic(x_train, y_train)
        predicted = predict_logistic(x_test, model)
        prior = np.full(len(test), (y_train.sum() + 0.5) / (len(train) + 1.0))
        results.append(dict(seed=seed, attempts=len(test), hits=int(y_test.sum()),
                            predicted=predicted.tolist(), prior=prior.tolist(),
                            observed=y_test.tolist()))
    observed = np.array([y for item in results for y in item['observed']])
    learned = np.array([p for item in results for p in item['predicted']])
    prior = np.array([p for item in results for p in item['prior']])
    def metrics(probabilities):
        p = np.clip(probabilities, 1e-6, 1 - 1e-6)
        return dict(brier=float(np.mean((p - observed)**2)),
                    log_loss=float(np.mean(-observed * np.log(p) -
                                           (1 - observed) * np.log1p(-p))))
    return dict(schema='route_capture_ml_group_holdout_v1',
                fields=len(results), attempts=len(observed), hits=int(observed.sum()),
                feature_names=FEATURE_NAMES, prior=metrics(prior),
                learned=metrics(learned), folds=results,
                agent_connected=False,
                caveat='Executed first legs only; unflown alternatives unlabeled')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive', type=Path)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    records = load_audited_attempts(args.archive)
    result = held_out_report(records)
    full_model = fit_regularized_logistic(
        np.array([row['features'] for row in records]),
        np.array([row['hit'] for row in records]))
    result['full_fit_prototype'] = dict(
        center=full_model[0].tolist(), scale=full_model[1].tolist(),
        coefficients=full_model[2].tolist(),
        role='offline diagnostic only; not a flight policy')
    result['source_archive_sha256'] = hashlib.sha256(
        args.archive.read_bytes()).hexdigest()
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({key: result[key] for key in
                      ('fields', 'attempts', 'hits', 'prior', 'learned')},
                     indent=2))


if __name__ == '__main__':
    main()
