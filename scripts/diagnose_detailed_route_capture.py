"""Audited, seed-held-out test of richer observation-only capture features.

This is an offline diagnostic, not a flight policy.  It learns only from
executed first-leg outcomes and never labels unflown routes.  The incumbent
six-feature model is evaluated on the *same* attempts as the detailed model.
"""

import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from scripts.diagnose_route_capture_ml import (
    FEATURE_NAMES, first_leg_features, held_out_report, load_audited_attempts,
)


DETAIL_NAMES = FEATURE_NAMES + (
    'remaining_burn_after_deadline',
    'relative_altitude',
    'relative_lateral_speed',
    'nearby_10m_fraction',
    'nearby_25m_fraction',
    'departure_velocity_mismatch',
    'initial_force_change_fraction',
)


def detailed_first_leg_features(decision):
    """Add route/actuator context available before the chosen leg executes.

    The last two quantities are proxies, not simulated actuator dynamics:
    arrival velocity is the planner's committed derivative and the required
    force comes from a constant-acceleration intercept approximation.
    """
    base = first_leg_features(decision)
    target = int(decision['target'])
    node = np.asarray(decision['features']['nodes'][target], dtype=float)
    context = np.asarray(decision['features']['context'], dtype=float)
    states = np.asarray(decision['visible_states'], dtype=float)
    acceleration = np.asarray(decision['acceleration'], dtype=float)
    horizon = float(decision['deadline'] - decision['time'])
    ends_v = decision.get('ends_v', ())
    if (node.shape != (12,) or context.shape != (9,) or
            states.ndim != 2 or states.shape[1] != 6 or
            not 0 <= target < len(states) or acceleration.shape != (3,) or
            not decision.get('route_metadata_aligned', False) or
            not ends_v or horizon <= 0):
        raise ValueError('Detailed features require an aligned visible first leg')
    end_v = np.asarray(ends_v[0], dtype=float)
    if end_v.shape != (3,) or not np.all(np.isfinite(end_v)):
        raise ValueError('Invalid planned departure velocity')
    relative = node[:3] * 150.0
    relative_v = node[3:6] * 30.0
    available = max(float(context[8] * 20.0), 1e-6)
    required_acceleration = 2.0 * (relative + relative_v * horizon) / horizon**2
    force_change = (required_acceleration - acceleration) / available
    extra = np.array((
        context[6] - horizon / 30.0,
        node[2],
        np.linalg.norm(relative_v[:2]) / 30.0,
        node[9],
        node[10],
        np.linalg.norm(end_v - states[target, 3:6]) / 30.0,
        np.linalg.norm(force_change),
    ), dtype=float)
    result = np.r_[base, extra]
    if result.shape != (len(DETAIL_NAMES),) or not np.all(np.isfinite(result)):
        raise ValueError('Non-finite detailed capture features')
    return result


def load_detailed_audited_attempts(archive_path):
    """Read audited executed outcomes without editing collector dependencies."""
    records = []
    with ZipFile(archive_path) as archive:
        folders = sorted(name[:-len('/audit.json')]
                         for name in archive.namelist()
                         if name.startswith('data/') and
                         name.endswith('/audit.json'))
        if not folders:
            raise ValueError('No audited fields')
        for folder in folders:
            seed = int(folder.split('/')[-1])
            audit = json.loads(archive.read(folder + '/audit.json'))
            gate = json.loads(archive.read(folder + '/gate.json'))
            raw = archive.read(folder + '/decisions.jsonl')
            if (not audit['parity_verified'] or
                    not audit['all_pop_events_complete'] or
                    not gate['exact_parity'] or
                    hashlib.sha256(raw).hexdigest() != audit['dataset_sha256']):
                raise ValueError(f'Unaudited or changed stream on {seed}')
            first, outcomes = {}, {}
            for line in raw.splitlines():
                row = json.loads(line)
                if row['kind'] == 'decision':
                    first.setdefault(int(row['attempt_id']), row)
                elif row['kind'] == 'outcome':
                    attempt = int(row['attempt_id'])
                    if attempt in outcomes:
                        raise ValueError(f'Duplicate outcome on {seed}')
                    outcomes[attempt] = row
            if len(outcomes) != audit['executed_attempts']:
                raise ValueError(f'Incomplete outcomes on {seed}')
            for attempt, outcome in sorted(outcomes.items()):
                if outcome['outcome'] not in ('hit', 'miss'):
                    continue
                decision = first.get(attempt)
                if decision is None or decision['target'] != outcome['target']:
                    raise ValueError(f'Unmatched chosen leg on {seed}')
                records.append(dict(seed=seed, attempt_id=attempt,
                                    features=detailed_first_leg_features(decision),
                                    hit=int(outcome['outcome'] == 'hit')))
    return records


def compare_archives(archives):
    ordinary = []
    detailed = []
    for archive in archives:
        ordinary += load_audited_attempts(archive)
        detailed += load_detailed_audited_attempts(archive)
    keys = lambda rows: [(x['seed'], x['attempt_id'], x['hit']) for x in rows]
    if keys(ordinary) != keys(detailed) or len({x['seed'] for x in ordinary}) < 3:
        raise ValueError('Baseline and detailed audited attempts differ')
    baseline = held_out_report(ordinary)
    candidate = held_out_report(detailed)
    candidate['feature_names'] = DETAIL_NAMES
    # Exploratory feature ablation on these development fields only.  It is
    # never interpreted as independent validation of the winning feature.
    additions = []
    for column in range(len(FEATURE_NAMES), len(DETAIL_NAMES)):
        subset = [dict(row, features=np.r_[row['features'][:len(FEATURE_NAMES)],
                                           row['features'][column]])
                  for row in detailed]
        result = held_out_report(subset)
        wins = 0
        for prior_fold, candidate_fold in zip(
                baseline['folds'], result['folds']):
            if prior_fold['seed'] != candidate_fold['seed']:
                raise ValueError('Feature ablation changed field order')
            y = np.asarray(prior_fold['observed'], dtype=float)
            old = np.asarray(prior_fold['predicted'], dtype=float)
            new = np.asarray(candidate_fold['predicted'], dtype=float)
            wins += int(np.mean((new - y) ** 2) < np.mean((old - y) ** 2))
        additions.append(dict(
            feature=DETAIL_NAMES[column],
            fields_with_lower_brier=wins,
            brier=result['learned']['brier'],
            log_loss=result['learned']['log_loss'],
            brier_gain=baseline['learned']['brier'] - result['learned']['brier'],
            log_loss_gain=baseline['learned']['log_loss'] - result['learned']['log_loss'],
        ))
    return dict(
        schema='detailed_capture_group_holdout_v1',
        archives=[str(Path(p)) for p in archives],
        fields=baseline['fields'], attempts=baseline['attempts'],
        baseline=baseline, detailed=candidate,
        exploratory_single_additions=additions,
        brier_gain=baseline['learned']['brier'] - candidate['learned']['brier'],
        log_loss_gain=baseline['learned']['log_loss'] - candidate['learned']['log_loss'],
        agent_connected=False,
        caveat='Executed first legs only; no route-score or counterfactual claim',
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archives', type=Path, nargs='+')
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    report = compare_archives(args.archives)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({key: report[key] for key in (
        'fields', 'attempts', 'brier_gain', 'log_loss_gain')}, indent=2))


if __name__ == '__main__':
    main()
