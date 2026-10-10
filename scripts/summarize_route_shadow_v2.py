"""Read-only summary of parity-approved Scenario-4 route-shadow data fields."""
import argparse
import json
from pathlib import Path
from statistics import mean, median

from scripts.audit_route_shadow import audit
from scripts.collect_route_shadow import paired_gate


def summarize(directory, *, require_complete=True):
    directory = Path(directory)
    plan = json.loads((directory/'plan.json').read_text(encoding='utf-8'))
    planned = plan['seeds']
    if plan.get('scenario') != 4 or plan.get('event_schema') != 'route_shadow_events_v2':
        raise ValueError('Expected Scenario-4 v2 collection plan')
    if len(planned) != len(set(planned)):
        raise ValueError('Duplicate planned field seed')
    progress = directory/('summary.json' if require_complete else 'progress.json')
    state = json.loads(progress.read_text(encoding='utf-8'))
    completed = state['completed']
    if [r['seed'] for r in completed] != planned[:len(completed)]:
        raise ValueError('Completion order differs from predeclared plan')
    if require_complete and (not state['all_passed'] or len(completed) != len(planned)):
        raise ValueError('Collection is incomplete')
    rows = []
    for result in completed:
        seed = result['seed']
        if not result['exact_parity'] or not paired_gate(result['baseline'], result['shadow']):
            raise ValueError(f'Action/observation parity failed for seed {seed}')
        checked = audit(directory/str(seed))
        if not checked['all_pop_events_complete'] or checked['dataset_sha256'] != result['dataset_sha256']:
            raise ValueError(f'Incomplete/changed pop stream for seed {seed}')
        pop = checked['pop_counts']
        rows.append(dict(seed=seed, score=result['shadow']['score'],
                         observed_pops=pop['observed_pops'],
                         incidental_pops=pop['incidental_pops'],
                         unassigned_pops=pop['unassigned_pops'],
                         active_target_pops=pop['active_target_pops'],
                         attempts=checked['executed_attempts'],
                         decisions=checked['decisions'],
                         deadline_misses=checked['outcomes'].get('miss', 0),
                         dataset_sha256=checked['dataset_sha256']))
    scores = [r['score'] for r in rows]
    return dict(scenario=4, planned_fields=len(planned), completed_fields=len(rows),
                complete=len(rows) == len(planned) and bool(state.get('all_passed')),
                scores=scores, mean_score=mean(scores) if scores else None,
                median_score=median(scores) if scores else None,
                minimum_score=min(scores) if scores else None,
                maximum_score=max(scores) if scores else None,
                zero_pop_fields=sum(s == 0 for s in scores),
                total_observed_pops=sum(r['observed_pops'] for r in rows),
                total_incidental_pops=sum(r['incidental_pops'] for r in rows),
                total_unassigned_pops=sum(r['unassigned_pops'] for r in rows),
                total_attempts=sum(r['attempts'] for r in rows),
                total_decisions=sum(r['decisions'] for r in rows),
                source='fresh official flights; paired exact action/observation parity; '
                       'all observed pops, no counterfactual route labels',
                fields=rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--partial', action='store_true', help='Allow an unfinished progress file')
    args = parser.parse_args()
    print(json.dumps(summarize(args.directory, require_complete=not args.partial), indent=2))
