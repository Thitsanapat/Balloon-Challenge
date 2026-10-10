"""Audit the isolated development flights, without changing any live file."""
import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INCUMBENT_SHA256 = '9ff7f93d21a945d0479289be57bd893d8f33ab169fb1a6bfeb153ea442b6efb9'


def summarize(flights):
    scores = {0: 6, 2: 4}
    if len(flights) != 2 or {r.get('seed') for r in flights} != set(scores):
        raise ValueError('Exactly the two declared development fields required')
    pairs = []
    for flight in flights:
        if (flight.get('scenario') != 4 or not flight.get('terminated') or
                flight.get('truncated') or
                not flight.get('agent_file_unchanged_during_run') or
                not flight.get('agent_sources_unchanged_during_run') or
                flight.get('config', {}).get('agent_class_name') !=
                    'Scenario4CrossDepthValueAgent'):
            raise ValueError('Incomplete, changed, or wrong-policy official flight')
        seed = flight['seed']
        score = flight['score']
        if type(score) is not int or score < 0 or score != len(flight['pop_events']):
            raise ValueError('Score inconsistent with observed unique pops')
        d = flight['diagnostics']
        pairs.append(dict(seed=seed, incumbent=scores[seed], candidate=score,
                          delta=score-scores[seed],
                          wall_seconds=flight['wall_seconds'],
                          post_reset_wall_seconds=flight['post_reset_wall_seconds'],
                          value_routes_adopted=d.get('value_routes_adopted', 0),
                          value_shorter_adopted=d.get('value_shorter_adopted', 0),
                          value_pool_peak=d.get('value_pool_peak', 0)))
    pairs.sort(key=lambda r: r['seed'])
    deltas = [r['delta'] for r in pairs]
    return dict(scope='Development screen, not a qualification submission',
                pairs=pairs, incumbent_total=sum(scores.values()),
                candidate_total=sum(r['candidate'] for r in pairs),
                fresh_holdout_eligible=(sum(deltas) >= 1 and min(deltas) >= -1
                                       and any(d > 0 for d in deltas)),
                live_agent_changed=False, mpcc_enabled=False,
                note='No automatic promotion; calibrated shadow capture gate failed.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    flights = json.loads(args.report.read_text(encoding='utf-8'))
    result = summarize(flights)
    source = ROOT/'BalloonPoppingGymEnv/agents/scenario4_crossdepth_value_agent.py'
    incumbent = ROOT/'BalloonPoppingGymEnv/agents/submission_scenario4_pure_joint_v2.py'
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    if digest(incumbent) != INCUMBENT_SHA256:
        raise RuntimeError('The selected standalone changed; review before claiming preservation')
    if any(f['agent_sha256'] != digest(source) for f in flights):
        raise RuntimeError('Candidate source differs from the scored source')
    model = ROOT/flights[0]['config']['agent_kwargs']['model_path']
    result['sha256'] = dict(incumbent=digest(incumbent), candidate=digest(source),
                            frozen_capture_model=digest(model), raw_report=digest(args.report))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
