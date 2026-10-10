"""Validate a parity-approved dataset; no training or flight-policy changes.

Only executed first-leg attempts receive outcomes. Revisions are grouped, not
treated as independent trials. Cancellation/censoring is not a miss. Prefixes
are unlabelled proposal descriptions, not counterfactual performance evidence.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

from BalloonPoppingGymEnv.agents.route_attention_features import ROUTE_FEATURE_SCHEMA
from BalloonPoppingGymEnv.training.route_shadow import EVENT_SCHEMA
from scripts.collect_route_shadow import paired_gate


def audit_rows(rows):
    attempts = {}
    decisions = set()
    previous_time = -np.inf
    counts = Counter()
    prefix_counts = Counter()
    started = ended = False
    legacy = None
    initial_pops = set()
    pops = {}
    pop_relations = Counter()
    end_counts = None
    balloon_count = None
    for row in rows:
        if legacy is None:
            legacy = row['kind'] != 'episode_start'
        if ended:
            raise ValueError('Record after episode end')
        now = row['time']
        if not np.isfinite(now) or now < previous_time:
            raise ValueError('Nonfinite or out-of-order event time')
        previous_time = now
        if row['kind'] == 'episode_start':
            if started or legacy or row['event_schema'] != EVENT_SCHEMA:
                raise ValueError('Invalid episode start/schema')
            balloon_count = row['balloon_count']
            ids = row['initial_popped_ids']
            if (type(balloon_count) is not int or balloon_count < 0
                    or len(set(ids)) != len(ids)
                    or any(type(i) is not int or not 0 <= i < balloon_count for i in ids)):
                raise ValueError('Invalid initial popped IDs')
            initial_pops = set(ids)
            started = True
            continue
        if row['kind'] == 'pop':
            target = row['balloon_id']
            if (not started or type(target) is not int or not 0 <= target < balloon_count
                    or target in initial_pops or target in pops or row['event_id'] != len(pops)):
                raise ValueError('Invalid/duplicate pop event')
            active_id = row['active_attempt_id']
            pending = {i: a for i, a in attempts.items() if a['outcome'] is None}
            if active_id is None:
                if pending or row['active_target'] is not None:
                    raise ValueError('Pop incorrectly marked unassigned')
                expected = 'unassigned'
            else:
                if active_id not in pending or pending[active_id]['target'] != row['active_target']:
                    raise ValueError('Pop associated with a nonexistent/closed attempt')
                expected = 'active_target' if target == row['active_target'] else 'incidental'
            if row['relation'] != expected:
                raise ValueError('Incorrect pop relation')
            pops[target] = row
            pop_relations[expected] += 1
            continue
        if row['kind'] == 'episode_end':
            ids = row['final_popped_ids']
            if (not started or len(set(ids)) != len(ids)
                    or set(ids) != initial_pops | set(pops)):
                raise ValueError('Final popped IDs do not match observed events')
            end_counts = row['counts']
            ended = True
            continue
        attempt_id = row['attempt_id']
        if type(attempt_id) is not int or attempt_id < 0:
            raise ValueError('Invalid attempt ID')
        if row['kind'] == 'decision':
            if row['feature_schema'] != ROUTE_FEATURE_SCHEMA:
                raise ValueError('Unknown feature schema')
            if 'seed' in row or 'scenario' in row:
                raise ValueError('Provenance belongs outside training features')
            decision_id = row['decision_id']
            if decision_id in decisions:
                raise ValueError('Duplicate decision ID')
            decisions.add(decision_id)
            f = row['features']
            nodes = np.asarray(f['nodes'], dtype=float)
            context = np.asarray(f['context'], dtype=float)
            visible = np.asarray(f['visible_mask'])
            candidate = np.asarray(f['candidate_mask'])
            states = np.asarray(row['visible_states'], dtype=float)
            if (nodes.ndim != 2 or nodes.shape[1] != 12 or context.shape != (9,)
                    or visible.shape != (len(nodes),) or candidate.shape != visible.shape
                    or visible.dtype != np.bool_ or candidate.dtype != np.bool_
                    or states.shape != (len(nodes), 6)
                    or not all(np.all(np.isfinite(a)) for a in (nodes, context, states))):
                raise ValueError('Invalid scene features')
            if started and len(nodes) != balloon_count:
                raise ValueError('Scene size changed')
            if np.any(candidate & ~visible) or np.any(nodes[~visible]) or np.any(states[~visible]):
                raise ValueError('Hidden rows must be masked and zeroed')
            target = row['target']
            if (type(target) is not int or not 0 <= target < len(nodes)
                    or not visible[target] or not np.isfinite(row['deadline'])
                    or row['deadline'] <= now):
                raise ValueError('Invalid executed target/deadline')
            if bool(row['imitation_eligible']) != bool(candidate[target]):
                raise ValueError('Masked target cannot be a categorical imitation label')
            attempt = attempts.setdefault(attempt_id, dict(target=target, decisions=[], outcome=None))
            if attempt['target'] != target or attempt['outcome'] is not None:
                raise ValueError('Attempt target changed or attempt already ended')
            if any(i != attempt_id and a['outcome'] is None for i, a in attempts.items()):
                raise ValueError('Overlapping active attempts')
            attempt['decisions'].append(decision_id)
            if len(attempt['decisions']) == 1:
                attempt['imitation_eligible'] = row['imitation_eligible']
            attempt['deadline'] = row['deadline']
            prefixes = row['route_prefixes']
            if not row['route_metadata_aligned'] and prefixes:
                raise ValueError('Stale route cannot be a proposal')
            for prefix in prefixes:
                ids, durations = prefix['targets'], prefix['durations']
                if (prefix['actual_hits'] is not None or not ids or ids[0] != target
                        or len(ids) != len(durations) or len(set(ids)) != len(ids)
                        or any(type(i) is not int or not 0 <= i < len(nodes) for i in ids)
                        or not np.all(np.isfinite(durations)) or min(durations) <= 0):
                    raise ValueError('Invalid or falsely labelled unexecuted prefix')
                prefix_counts[len(ids)] += 1
        elif row['kind'] == 'outcome':
            if attempt_id not in attempts:
                raise ValueError('Outcome without executed attempt')
            attempt = attempts[attempt_id]
            outcome = row['outcome']
            if (outcome not in ('hit', 'miss', 'cancelled', 'censored')
                    or attempt['outcome'] is not None or row['target'] != attempt['target']
                    or row['deadline'] != attempt['deadline']):
                raise ValueError('Invalid or duplicate outcome')
            if outcome == 'miss' and now < row['deadline'] - 1e-9:
                raise ValueError('Premature deadline-miss label')
            if outcome == 'censored' and now >= row['deadline']:
                raise ValueError('Censoring must precede the deadline')
            if started and outcome == 'hit':
                event = pops.get(row['target'], {})
                if (event.get('active_attempt_id') != attempt_id or event.get('time') != now
                        or event.get('relation') != 'active_target'):
                    raise ValueError('Hit label without matching observed pop')
            attempt['outcome'] = outcome
            counts[outcome] += 1
        else:
            raise ValueError('Unknown record type')
    if any(a['outcome'] is None for a in attempts.values()):
        raise ValueError('Incomplete episode: unresolved attempts')
    event_counts = dict(initial_pops=len(initial_pops), observed_pops=len(pops),
                       active_target_pops=pop_relations['active_target'],
                       incidental_pops=pop_relations['incidental'],
                       unassigned_pops=pop_relations['unassigned'])
    if started:
        expected_counts = dict(decisions=len(decisions), **event_counts,
                               **{k: counts[k] for k in ('hit', 'miss', 'cancelled', 'censored')})
        if not ended or end_counts != expected_counts or counts['hit'] != event_counts['active_target_pops']:
            raise ValueError('Missing/invalid episode end or inconsistent event counters')
    return dict(decisions=len(decisions), executed_attempts=len(attempts),
                event_schema=EVENT_SCHEMA if started else 'legacy_first_leg_only',
                all_pop_events_complete=started and ended,
                pop_counts=event_counts if started else None,
                outcomes=dict(counts), prefix_length_histogram=dict(prefix_counts),
                # First decision only for imitation: no overweighting refreshes.
                imitation_decision_ids=[a['decisions'][0] for a in attempts.values()
                                        if a['imitation_eligible']],
                deadline_outcome_attempt_ids=[i for i, a in attempts.items()
                                              if a['outcome'] in ('hit', 'miss')],
                unresolved_counterfactuals='all route prefixes remain unlabelled')


def audit(directory):
    directory = Path(directory)
    gate = json.loads((directory/'gate.json').read_text())
    if not gate['exact_parity'] or not paired_gate(gate['baseline'], gate['shadow']):
        raise ValueError('Exact flight parity did not pass')
    path = directory/'decisions.jsonl'
    if hashlib.sha256(path.read_bytes()).hexdigest() != gate['dataset_sha256']:
        raise ValueError('Dataset hash differs from completed parity gate')
    with path.open(encoding='utf-8') as file:
        report = audit_rows(json.loads(line) for line in file if line.strip())
    counts = gate['shadow']['collection_counts']
    if (counts['decisions'] != report['decisions']
            or any(counts[k] != report['outcomes'].get(k, 0)
                   for k in ('hit', 'miss', 'cancelled', 'censored'))):
        raise ValueError('Dataset and flight counters disagree')
    if report['all_pop_events_complete']:
        if (any(counts.get(k) != v for k, v in report['pop_counts'].items())
                or report['pop_counts']['initial_pops'] + report['pop_counts']['observed_pops']
                != gate['shadow']['score']):
            raise ValueError('All-pop counters do not match official flight score')
    # Zero-pop / no-plan flights are valid evidence, not grounds to drop a seed.
    return dict(parity_verified=True, dataset_sha256=gate['dataset_sha256'], **report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.directory), indent=2))
