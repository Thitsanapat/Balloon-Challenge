"""Replay the multi-target agent's lock/switch decisions from a trajectory."""

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("trajectory", type=Path)
    parser.add_argument("--launch-time", type=float, default=4.0)
    parser.add_argument("--timeout", type=float, default=8.0)
    parser.add_argument("--cooldown", type=float, default=4.0)
    args = parser.parse_args()

    records = json.loads(args.trajectory.read_text(encoding="utf-8"))
    target = None
    acquired_at = 0.0
    minimum_range = np.inf
    failed_until = {}

    for record in records:
        now = float(record["time"])
        if now < args.launch_time:
            continue
        states = np.asarray(record["balloon_states"], dtype=float)
        status = np.asarray(record["balloon_status"], dtype=int)
        rocket = np.asarray(record["rocket_states"], dtype=float)
        position = rocket[:3] if np.isfinite(rocket[:3]).all() else np.array([0, 0, 20])
        velocity = rocket[3:6] if np.isfinite(rocket[3:6]).all() else np.zeros(3)

        keep = False
        reason = "unavailable"
        if target is not None and status[target] == 1:
            relative = states[target, :3] - position
            distance = np.linalg.norm(relative)
            closing = np.dot(relative, velocity) > 0.0
            minimum_range = min(minimum_range, distance)
            timed_out = now - acquired_at > args.timeout
            passed = not closing and distance > minimum_range + 4.0
            keep = not timed_out and not passed
            reason = "timeout" if timed_out else "passed"
        elif target is not None and status[target] == 2:
            reason = "popped"

        if keep:
            continue
        if target is not None and status[target] == 1:
            failed_until[target] = now + args.cooldown
        released = np.flatnonzero(status == 1)
        if released.size == 0:
            target = None
            continue
        available = np.asarray(
            [index for index in released if failed_until.get(int(index), 0.0) <= now],
            dtype=int,
        )
        if available.size == 0:
            available = released
        relative = states[available, :3] - position
        distance = np.linalg.norm(relative, axis=1)
        if np.linalg.norm(velocity) > 3.0:
            direction = velocity / np.linalg.norm(velocity)
            alignment = relative @ direction / np.maximum(distance, 1e-9)
            score = distance * (1.4 - 0.4 * alignment)
        else:
            score = distance
        old_target = target
        target = int(available[np.argmin(score)])
        acquired_at = now
        minimum_range = np.inf
        print(
            f"{now:6.2f}s: {old_target} -> {target} ({reason}), "
            f"range={distance[np.argmin(score)]:.1f}m"
        )


if __name__ == "__main__":
    main()
