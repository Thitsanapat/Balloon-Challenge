"""Print a compact score and nearest-miss summary for a trajectory JSON file."""

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("trajectory", type=Path)
    parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args()

    records = json.loads(args.trajectory.read_text(encoding="utf-8"))
    times = np.asarray([record["time"] for record in records], dtype=float)
    rocket = np.asarray([record["rocket_states"][:3] for record in records], dtype=float)
    balloons = np.asarray([record["balloon_states"] for record in records], dtype=float)
    status = np.asarray([record["balloon_status"] for record in records], dtype=int)

    flyable = np.isfinite(rocket).all(axis=1)
    distances = np.linalg.norm(balloons[:, :, :3] - rocket[:, None, :], axis=2)
    distances[~flyable] = np.inf
    distances[status == 0] = np.inf
    closest_by_balloon = distances.min(axis=0)
    closest_indices = np.argsort(closest_by_balloon)[: args.limit]

    print(f"file: {args.trajectory.name}")
    print(f"steps: {len(records)}, final time: {times[-1]:.2f} s")
    print(f"popped: {np.count_nonzero(status[-1] == 2)}")
    previous_status = np.vstack((status[0], status[:-1]))
    pop_steps, pop_balloons = np.where((status == 2) & (previous_status != 2))
    if pop_steps.size:
        print(
            "pop events: "
            + ", ".join(
                f"#{balloon}@{times[step]:.2f}s"
                f"({np.linalg.norm(np.asarray(records[step]['rocket_states'][3:6], dtype=float)):.1f}m/s)"
                for step, balloon in zip(pop_steps, pop_balloons)
            )
        )
    print("closest eligible balloons:")
    for index in closest_indices:
        step = int(np.argmin(distances[:, index]))
        delta = balloons[step, index, :3] - rocket[step]
        print(
            f"  #{index}: {closest_by_balloon[index]:.3f} m at {times[step]:.2f} s; "
            f"target-minus-rocket=({delta[0]:.2f}, {delta[1]:.2f}, {delta[2]:.2f}); "
            f"rocket=({rocket[step, 0]:.2f}, {rocket[step, 1]:.2f}, {rocket[step, 2]:.2f})"
        )


if __name__ == "__main__":
    main()
