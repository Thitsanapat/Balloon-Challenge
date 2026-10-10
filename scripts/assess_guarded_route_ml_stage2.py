"""Read-only integrity and promotion-screen report for the frozen stage-2 holdout.

This analysis is not imported by either flight agent and never changes a
submission. Incomplete experiments have no pass/fail verdict.
"""

import argparse
import json
from pathlib import Path


EXPECTED_SEEDS = (
    140200341, 1098270439, 1904235602, 568300771,
    178420909, 820431975, 1126043508, 662504311,
)
POLICIES = ('selected', 'learned')


def assess(data):
    plan = data.get('plan')
    if not isinstance(plan, dict) or tuple(plan.get('seeds', ())) != EXPECTED_SEEDS:
        raise ValueError('Not the frozen eight-seed stage-2 plan')
    if tuple(plan.get('order', ())) != POLICIES:
        raise ValueError('Unexpected policy order')
    configs = plan.get('configs') or {}
    if set(configs) != set(POLICIES):
        raise ValueError('Missing frozen policy metadata')
    flights = data.get('flights')
    if not isinstance(flights, list):
        raise ValueError('Missing flight list')
    outcomes = {}
    for entry in flights:
        seed, policy = entry['seed'], entry['policy']
        if seed not in EXPECTED_SEEDS or policy not in POLICIES:
            raise ValueError('Unexpected seed or policy in flight')
        if (seed, policy) in outcomes:
            raise ValueError('Duplicate flight')
        result = entry['result']
        if (result.get('seed') != seed or result.get('scenario') != 4 or
                result.get('agent_sha256') != configs[policy].get('agent_sha256') or
                not result.get('terminated') or result.get('truncated') or
                not result.get('agent_sources_unchanged_during_run')):
            raise ValueError('Official/source gate failed for a completed flight')
        score = result.get('score')
        if type(score) is not int or score < 0 or score != len(result.get('pop_events', [])):
            raise ValueError('Score and pop events disagree')
        diagnostics = result.get('diagnostics') or {}
        tracking = diagnostics.get('tracking_steps', 0)
        outcomes[(seed, policy)] = dict(
            score=score,
            popped_targets=[int(event[0]) for event in result['pop_events']],
            mean_tracking_error_m=(diagnostics.get('position_error_sum', 0) /
                                   tracking if tracking else None),
            allocator_saturation_fraction=(diagnostics.get('saturated_steps', 0) /
                                           tracking if tracking else None),
        )
    rows = []
    for seed in EXPECTED_SEEDS:
        if (seed, 'selected') not in outcomes or (seed, 'learned') not in outcomes:
            continue
        selected, learned = outcomes[(seed, 'selected')], outcomes[(seed, 'learned')]
        rows.append(dict(seed=seed, selected=selected, learned=learned,
                         score_delta=learned['score'] - selected['score']))
    complete = bool(data.get('complete'))
    if complete and len(outcomes) != 2 * len(EXPECTED_SEEDS):
        raise ValueError('Report claims complete with missing flights')
    report = dict(complete=complete, completed_flights=len(outcomes),
                  required_flights=2 * len(EXPECTED_SEEDS), paired=rows,
                  conclusion='pending')
    if not complete:
        return report
    selected_total = sum(row['selected']['score'] for row in rows)
    learned_total = sum(row['learned']['score'] for row in rows)
    no_regression_count = sum(row['score_delta'] >= 0 for row in rows)
    worst_delta = min(row['score_delta'] for row in rows)
    passes = (learned_total - selected_total >= 4 and worst_delta >= -2 and
              no_regression_count >= 4)
    report.update(totals=dict(selected=selected_total, learned=learned_total),
                  aggregate_delta=learned_total - selected_total,
                  no_regression_count=no_regression_count,
                  worst_delta=worst_delta,
                  conclusion='screen_passed' if passes else 'screen_failed',
                  automatic_promotion=False)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('result', type=Path)
    args = parser.parse_args()
    print(json.dumps(assess(json.loads(args.result.read_text(encoding='utf-8'))),
                     indent=2))


if __name__ == '__main__':
    main()
