"""Offline attribution of sampled near-misses from guidance telemetry.

Truth position, when present, comes from post-step simulator info inside a
development report. This module is never loaded by the competition agent.
The vector projections describe error contributions, not causal effects of a
counterfactual controller change.
"""

import argparse
import json
from pathlib import Path

import numpy as np


VECTOR_NAMES = ('tracking_vector', 'alignment_vector',
                'prediction_error_vector')


def summarize_flight(flight, radius=1.5, near_limit=5.0):
    telemetry = flight.get('guidance_telemetry') or {}
    missed = {}
    for episode in telemetry.get('target_episodes', []):
        if episode.get('end_reason') == 'popped':
            continue
        closest = episode.get('true_closest')
        if closest is None or not radius < closest['distance'] <= near_limit:
            continue
        target = int(episode['target'])
        if target in missed and missed[target]['distance'] <= closest['distance']:
            continue
        vectors = [np.asarray(closest[name], dtype=float) for name in VECTOR_NAMES]
        if any(vec.shape != (3,) or not np.all(np.isfinite(vec)) for vec in vectors):
            raise ValueError('Invalid sampled error vectors')
        residual = sum(vectors)
        distance = float(np.linalg.norm(residual))
        if abs(distance - closest['distance']) > 1e-5:
            raise ValueError('Error decomposition does not close')
        direction = residual / max(distance, 1e-9)
        projected = {name.removesuffix('_vector') + '_along_miss_m':
                     float(np.dot(vec, direction))
                     for name, vec in zip(VECTOR_NAMES, vectors)}
        missed[target] = dict(target=target, distance=distance,
                              margin_outside_balloon_m=distance-radius,
                              time_to_deadline=closest['time_to_deadline'],
                              allocator_saturated_at_sample=bool(
                                  closest.get('command', {}).get(
                                      'allocator_saturated', False)),
                              **projected)
    return dict(seed=flight.get('seed'), score=flight['score'],
                near_misses=sorted(missed.values(),
                                   key=lambda row: row['distance']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('report', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    contents = json.loads(args.report.read_text(encoding='utf-8'))
    flights = contents if isinstance(contents, list) else [contents]
    result = [summarize_flight(flight) for flight in flights]
    rendered = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + '\n', encoding='utf-8')
    print(rendered)


if __name__ == '__main__':
    main()
