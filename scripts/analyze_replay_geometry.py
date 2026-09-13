"""Compare public/local replay geometry without using it inside an agent."""

import argparse
import json
from pathlib import Path

import numpy as np


def load_frames(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return payload["trajectories"] if isinstance(payload, dict) else payload


def analyze(path):
    frames = load_frames(path)
    time = np.array([frame["time"] for frame in frames], dtype=float)
    rocket = np.array([frame["rocket_states"] for frame in frames], dtype=float)
    status = np.array([frame["balloon_status"] for frame in frames], dtype=np.int8)
    position, velocity = rocket[:, :3], rocket[:, 3:6]
    speed = np.linalg.norm(velocity, axis=1)
    launch_candidates = np.flatnonzero(speed > .05)
    launch = int(launch_candidates[0]) if launch_candidates.size else 0
    steps = np.linalg.norm(np.diff(position, axis=0), axis=1)
    horizontal_steps = np.linalg.norm(np.diff(position[:, :2], axis=0), axis=1)
    # Smooth heading at 0.2 s spacing so integrator chatter is not counted as turns.
    stride = max(1, int(round(.2 / np.median(np.diff(time)))))
    sampled_v = velocity[launch::stride, :2]
    moving = np.linalg.norm(sampled_v, axis=1) > 2.
    headings = np.unwrap(np.arctan2(sampled_v[moving, 1], sampled_v[moving, 0]))
    heading_variation = np.degrees(np.sum(np.abs(np.diff(headings))))
    hit_counts = np.sum(status == 2, axis=1)
    hit_steps = np.flatnonzero(np.diff(hit_counts, prepend=0) > 0)
    hits = []
    previous = np.zeros(status.shape[1], dtype=np.int8)
    for step in hit_steps:
        indices = np.flatnonzero((status[step] == 2) & (previous != 2))
        hits.extend({"index": int(index), "time": round(float(time[step]), 2),
                     "z": round(float(position[step, 2]), 1),
                     "vz": round(float(velocity[step, 2]), 1),
                     "speed": round(float(speed[step]), 1)} for index in indices)
        previous = status[step].copy()
    powered = time <= time[launch] + 30.
    displacement = np.linalg.norm(position[-1] - position[launch])
    return {
        "file": str(path), "frames": len(frames), "launch_time": round(float(time[launch]), 2),
        "end_time": round(float(time[-1]), 2), "score": int(hit_counts[-1]),
        "path_length": round(float(np.sum(steps[launch:])), 1),
        "horizontal_path": round(float(np.sum(horizontal_steps[launch:])), 1),
        "path_to_displacement": round(float(np.sum(steps[launch:]) / max(displacement, 1e-9)), 2),
        "heading_variation_deg": round(float(heading_variation), 1),
        "apogee": round(float(np.max(position[launch:, 2])), 1),
        "apogee_time": round(float(time[launch + np.argmax(position[launch:, 2])]), 2),
        "burnout_z": round(float(position[np.flatnonzero(powered)[-1], 2]), 1),
        "burnout_vz": round(float(velocity[np.flatnonzero(powered)[-1], 2]), 1),
        "burnout_speed": round(float(speed[np.flatnonzero(powered)[-1]]), 1),
        "mean_powered_speed": round(float(np.mean(speed[powered & (np.arange(len(time)) >= launch)])), 1),
        "hits": hits,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("replays", nargs="+")
    args = parser.parse_args()
    print(json.dumps([analyze(path) for path in args.replays], indent=2))


if __name__ == "__main__":
    main()
