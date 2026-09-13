"""Offline beam search for dense, acceleration-feasible balloon routes.

This is a field-analysis tool, not an agent.  It treats the rocket as a point
mass and uses constant-acceleration legs.  Candidate routes still need to be
converted to a six-degree-of-freedom trajectory and verified in the simulator.
"""

import argparse
from dataclasses import dataclass

import numpy as np


@dataclass
class Node:
    step: int
    position: np.ndarray
    velocity: np.ndarray
    visited: int
    route: tuple
    arrivals: tuple
    effort: float


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("field")
    parser.add_argument("--start", type=float, default=36.3)
    parser.add_argument("--end", type=float, default=67.5)
    parser.add_argument("--beam", type=int, default=3000)
    parser.add_argument("--branch", type=int, default=18)
    parser.add_argument("--depth", type=int, default=24)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--min-leg", type=float, default=.45)
    parser.add_argument("--max-leg", type=float, default=5.)
    parser.add_argument("--max-thrust-accel", type=float, default=14.5)
    parser.add_argument("--max-speed", type=float, default=55.)
    parser.add_argument("--position", nargs=3, type=float, default=[0., 0., 20.])
    parser.add_argument("--velocity", nargs=3, type=float, default=[0., 0., 0.])
    parser.add_argument("--exclude", nargs="*", type=int, default=[])
    args = parser.parse_args()

    with np.load(args.field) as data:
        balloons = np.asarray(data["balloon_flights"][:, :3], dtype=float)
        releases = np.asarray(data["release_steps"], dtype=int)
        dt = float(data["time_step"])

    start_step = int(round(args.start / dt))
    end_step = min(int(round(args.end / dt)), balloons.shape[2] - 1)
    min_leg = max(1, int(round(args.min_leg / dt)))
    max_leg = max(min_leg, int(round(args.max_leg / dt)))
    gravity = np.array([0., 0., 9.80665])
    visited = sum(1 << int(index) for index in args.exclude)
    initial = Node(start_step, np.asarray(args.position, dtype=float),
                   np.asarray(args.velocity, dtype=float), visited,
                   (), (), 0.)
    beam = [initial]
    print(f"depth=0 beam=1 best_time={args.start:.2f}")

    for depth in range(1, args.depth + 1):
        children = []
        for node in beam:
            first = node.step + min_leg
            last = min(node.step + max_leg, end_step)
            if first > last:
                continue
            sample_steps = np.arange(first, last + 1, args.stride)
            leg_times = (sample_steps - node.step) * dt
            target_ids = np.array(
                [i for i in range(balloons.shape[0])
                 if not (node.visited >> i) & 1], dtype=int,
            )
            if not target_ids.size:
                continue
            # [time, target, xyz]
            targets = balloons[target_ids][:, :, sample_steps]
            targets = np.transpose(targets, (2, 0, 1))
            delta = targets - node.position[None, None, :]
            tau = leg_times[:, None, None]
            average_velocity = delta / tau
            # A collision does not require matching balloon velocity.  Retain
            # several fly-through velocities and test the endpoint acceleration
            # of the cubic Hermite leg joining (p0, v0) to (p1, v1).
            beta = np.array([-.5, 0., .5, 1.])
            end_velocity = (
                average_velocity[:, :, None, :]
                + beta[None, None, :, None]
                * (average_velocity - node.velocity[None, None, :])[:, :, None, :]
            )
            tau4 = tau[:, :, None, :]
            delta4 = delta[:, :, None, :]
            a2 = (3. * delta4
                  - (2. * node.velocity[None, None, None, :] + end_velocity) * tau4
                  ) / tau4**2
            a3 = (-2. * delta4
                  + (node.velocity[None, None, None, :] + end_velocity) * tau4
                  ) / tau4**3
            accel_start = 2. * a2
            accel_end = 2. * a2 + 6. * a3 * tau4
            thrust_accel = np.maximum(
                np.linalg.norm(accel_start + gravity, axis=3),
                np.linalg.norm(accel_end + gravity, axis=3),
            )
            speed = np.linalg.norm(end_velocity, axis=3)
            released = sample_steps[:, None] >= releases[target_ids][None, :]
            feasible = (released[:, :, None]
                        & (thrust_accel <= args.max_thrust_accel)
                        & (speed <= args.max_speed))
            local = []
            for column, target in enumerate(target_ids):
                rows, velocity_choices = np.nonzero(feasible[:, column, :])
                if not rows.size:
                    continue
                # Keep the earliest intercept and a lower-effort alternative.
                earliest = int(np.argmin(rows))
                economical = int(np.argmin(thrust_accel[rows, column, velocity_choices]))
                selected = [earliest]
                if economical != earliest:
                    selected.append(economical)
                for choice in selected:
                    row, velocity_choice = int(rows[choice]), int(velocity_choices[choice])
                    step = int(sample_steps[row])
                    effort = node.effort + float(thrust_accel[row, column, velocity_choice] ** 2
                                                 * leg_times[row])
                    local.append((step + .0002 * effort, target, row,
                                  velocity_choice, step, effort))
            local.sort(key=lambda item: item[0])
            for _, target, row, velocity_choice, step, effort in local[:args.branch]:
                column = int(np.flatnonzero(target_ids == target)[0])
                children.append(Node(
                    step,
                    targets[row, column].copy(),
                    end_velocity[row, column, velocity_choice].copy(),
                    node.visited | (1 << int(target)),
                    node.route + (int(target),),
                    node.arrivals + (step * dt,),
                    effort,
                ))
        if not children:
            print(f"depth={depth} no feasible continuation")
            break
        # Preserve distinct recent route histories; otherwise one easy target
        # and time bin can consume nearly the whole beam.
        children.sort(key=lambda node: (node.step + .0002 * node.effort,
                                        node.effort))
        beam, seen = [], set()
        for node in children:
            key = (node.route[-3:], node.step // args.stride)
            if key in seen:
                continue
            seen.add(key)
            beam.append(node)
            if len(beam) >= args.beam:
                break
        best = beam[0]
        route = " ".join(
            f"#{target}@{arrival:.2f}"
            for target, arrival in zip(best.route, best.arrivals)
        )
        print(f"depth={depth} beam={len(beam)} best_time={best.step*dt:.2f} "
              f"speed={np.linalg.norm(best.velocity):.1f} route={route}")


if __name__ == "__main__":
    main()
