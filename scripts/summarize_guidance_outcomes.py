"""Read-only analysis of saved metrics; never imported by a competition agent.

Route proposals overlap and are superseded. Distinct planned IDs are NOT
promised achievable hits; unexecuted proposals are not tracking failures.
"""
import argparse
import json
from pathlib import Path


def summarize(result):
    pops = {int(i) for i, _ in result['pop_events']}
    planned = {int(i) for _, route, _ in result.get('route_events', []) for i in route}
    selected = {int(i) for _, i in result.get('target_events', [])}
    diagnostics = result.get('diagnostics', {})
    n = diagnostics.get('tracking_steps', 0)
    episodes = (result.get('guidance_telemetry') or {}).get('target_episodes', [])
    missed = []
    for episode in episodes:
        sample = episode.get('closest')
        # Only report executed targeting episodes, not unexecuted route tails.
        if episode.get('end_reason') == 'popped' or sample is None:
            continue
        names = ('tracking_error', 'alignment_error', 'prediction_error')
        missed.append(dict(target=episode['target'], end_reason=episode.get('end_reason'),
            later_popped=int(episode['target']) in pops,
            closest_distance=sample['distance'], time_to_deadline=sample['time_to_deadline'],
            largest_component=max(names, key=lambda key: sample[key]),
            components={key: sample[key] for key in names}))
    return dict(seed=result['seed'], score=result['score'], truncated=result['truncated'],
        unique_planned=len(planned), unique_selected=len(selected),
        planned_but_never_selected=sorted(planned-selected),
        selected_not_popped=sorted(selected-pops), popped_not_planned=sorted(pops-planned),
        mean_sampled_tracking_error=diagnostics.get('position_error_sum', 0.)/n if n else None,
        # Saturation is counted even outside reference tracking: no shared
        # denominator is recorded, so do not manufacture a saturation fraction.
        saturated_control_steps=diagnostics.get('saturated_steps', 0),
        tracking_samples=n,
        maximum_planned_depth=diagnostics.get('maximum_depth'),
        telemetry_available=bool(result.get('guidance_telemetry')),
        unpopped_targeting_episodes=missed,
        scope='Distinct proposals are not score promises. Error-component ranking is diagnostic, not proof of causality.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('reports', nargs='+', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Refusing to overwrite analysis')
    rows = []
    for path in args.reports:
        for result in json.loads(path.read_text(encoding='utf-8-sig')):
            rows.append(dict(source=str(path), **summarize(result)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(rows, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
