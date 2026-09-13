"""Rank balloon encounters along an unpowered constant-gravity coast."""

import argparse
import json
from pathlib import Path

import numpy as np


GRAVITY = np.array([0.0, 0.0, -9.80665])


def vacuum_position(position, velocity, elapsed):
    elapsed = np.asarray(elapsed, dtype=float)
    return (
        np.asarray(position, dtype=float)
        + elapsed[..., None] * np.asarray(velocity, dtype=float)
        + 0.5 * elapsed[..., None] ** 2 * GRAVITY
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("field", type=Path)
    parser.add_argument("report", type=Path)
    parser.add_argument("--duration", type=float, default=7.0)
    parser.add_argument("--top", type=int, default=12)
    args = parser.parse_args()

    report = json.loads(args.report.read_text(encoding="utf-8"))
    final_pop = report["pop_states"][-1]
    start_time = float(final_pop["time"])
    position = np.asarray(final_pop["rocket_position"], dtype=float)
    velocity = np.asarray(final_pop["rocket_velocity"], dtype=float)
    popped = {int(item["index"]) for item in report["pop_states"]}

    with np.load(args.field) as cache:
        flights = np.asarray(cache["balloon_flights"])
        release_steps = np.asarray(cache["release_steps"], dtype=int)
        dt = float(cache["time_step"])
    start_step = int(round(start_time / dt))
    end_step = min(flights.shape[2] - 1, start_step + int(args.duration / dt))
    steps = np.arange(start_step, end_step + 1)
    elapsed = (steps - start_step) * dt
    rocket = vacuum_position(position, velocity, elapsed)

    ranked = []
    for index in range(flights.shape[0]):
        if index in popped:
            continue
        eligible = steps >= release_steps[index]
        if not np.any(eligible):
            continue
        distance = np.linalg.norm(
            flights[index, :3, steps] - rocket, axis=1
        )
        distance[~eligible] = np.inf
        local = int(np.argmin(distance))
        ranked.append((float(distance[local]), index, float(start_time + elapsed[local])))
    ranked.sort()
    for distance, index, time in ranked[: args.top]:
        print(f"#{index} distance={distance:.2f}m time={time:.2f}s")


if __name__ == "__main__":
    main()
