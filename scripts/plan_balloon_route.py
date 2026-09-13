"""Estimate high-throughput balloon routes from a recorded deterministic field."""

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("trajectory", type=Path)
    parser.add_argument("--start", type=float, default=15.0)
    parser.add_argument("--duration", type=float, default=35.0)
    parser.add_argument("--speed", type=float, default=25.0)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--position", nargs=3, type=float, default=[0.0, 0.0, 20.0])
    parser.add_argument("--exclude", nargs="*", type=int, default=[])
    args = parser.parse_args()

    payload = json.loads(args.trajectory.read_text(encoding="utf-8"))
    records = payload["trajectories"] if isinstance(payload, dict) else payload
    times = np.asarray([record["time"] for record in records])
    balloons = np.asarray([record["balloon_states"] for record in records])[:, :, :3]
    statuses = np.asarray([record["balloon_status"] for record in records])
    start_index = int(np.searchsorted(times, args.start))
    end_index = min(len(times) - 1, int(np.searchsorted(times, args.start + args.duration)))

    current_index = start_index
    position = np.asarray(args.position, dtype=float)
    visited = set(args.exclude)
    route = []
    while current_index < end_index:
        best = None
        for target in range(balloons.shape[1]):
            if target in visited:
                continue
            for sample in range(current_index + args.stride, end_index + 1, args.stride):
                if statuses[sample, target] != 1:
                    continue
                elapsed = times[sample] - times[current_index]
                distance = np.linalg.norm(balloons[sample, target] - position)
                if distance <= args.speed * elapsed:
                    candidate = (sample, distance, target)
                    if best is None or candidate < best:
                        best = candidate
                    break
        if best is None:
            break
        sample, distance, target = best
        route.append((target, times[sample], distance))
        visited.add(target)
        position = balloons[sample, target].copy()
        current_index = sample

    print(
        f"start={args.start:.1f}s duration={args.duration:.1f}s speed={args.speed:.1f}m/s "
        f"count={len(route)}"
    )
    print(" ".join(f"#{target}@{time:.1f}({distance:.1f}m)" for target, time, distance in route))


if __name__ == "__main__":
    main()
