"""Offline paired forecast check from live observations of an official episode.

This diagnostic never launches a rocket and never reads hidden balloon flights,
gust samples, a seed inside a policy, or prerecorded trajectories. It records
forecasts at one observation and verifies them against later observations from
the same official environment. A forecast result is not a pop-score result.
"""

import argparse
import copy
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from BalloonPoppingGymEnv.agents import scenario4_regularized_wind_agent
from BalloonPoppingGymEnv.agents import scenario4_wind_profile_agent
from BalloonPoppingGymEnv.agents.scenario4_regularized_wind_agent import (
    RegularizedObservedWindProfile,
)
from BalloonPoppingGymEnv.agents.scenario4_wind_profile_agent import (
    ObservedWindProfile,
)
from BalloonPoppingGymEnv.envs.balloon_world import BalloonPoppingEnv
from BalloonPoppingGymEnv.envs import balloon_world
from BalloonPoppingGymEnv.evaluation.evaluate import load_scenario_parameters


def summarize(errors, corrected):
    e = np.asarray(errors, dtype=float)
    return {
        "count": int(len(e)),
        "rmse_m": float(np.sqrt(np.mean(e ** 2))) if len(e) else None,
        "mean_m": float(np.mean(e)) if len(e) else None,
        "median_m": float(np.median(e)) if len(e) else None,
        "p90_m": float(np.quantile(e, .9)) if len(e) else None,
        "horizontal_within_1p5m_fraction": float(np.mean(e <= 1.5)) if len(e) else None,
        "corrected_fraction": float(corrected / len(e)) if len(e) else None,
    }


def evaluate_seed(seed, start=24.0, until=60.0, sample_interval=2.0,
                  horizons=(2.0, 4.0, 6.0, 8.0), blend=.75):
    """Compare constant velocity, incumbent and candidate on identical targets."""
    parameters, _ = load_scenario_parameters(4)
    parameters = copy.deepcopy(parameters)
    parameters["scenario"]["random_seed"] = int(seed)
    dt = float(parameters["simulation"]["time_step"])
    start_step = round(start / dt)
    until_step = round(until / dt)
    stride = round(sample_interval / dt)
    horizon_steps = [(float(h), round(h / dt)) for h in horizons]
    if (start_step < 0 or stride < 1 or until_step < start_step or
            not np.isfinite(blend) or not 0 <= blend <= 1 or
            not horizon_steps or any(h <= 0 or steps < 1 or
                                     abs(steps * dt - h) > 1e-8
                                     for h, steps in horizon_steps)):
        raise ValueError("Invalid forecast diagnostic sampling settings")

    incumbent = ObservedWindProfile()
    candidate = RegularizedObservedWindProfile()
    due = defaultdict(list)
    errors = defaultdict(lambda: defaultdict(list))
    corrected = defaultdict(lambda: defaultdict(int))
    changed_only = defaultdict(lambda: defaultdict(list))
    env = BalloonPoppingEnv(render_mode=None, parameters=parameters)
    no_launch = {"launch": False, "launch_inclination_heading": [90.0, 0.0],
                 "tvc": [0.0, 0.0], "throttle": 0.0, "roll": 0.0}
    try:
        obs, _ = env.reset(seed=int(seed))
        end_step = until_step + max(steps for _, steps in horizon_steps)
        for step in range(end_step + 1):
            if step in due:
                states = np.asarray(obs["balloon_states"], dtype=float)
                status = np.asarray(obs["balloon_status"]).reshape(-1)
                for index, horizon, predictions, nonzero in due.pop(step):
                    if status[index] != 1 or not np.all(np.isfinite(states[index])):
                        continue
                    truth = states[index, :2]
                    for name, position in predictions.items():
                        error = float(np.linalg.norm(position - truth))
                        errors[horizon][name].append(error)
                        corrected[horizon][name] += nonzero[name]
                        if nonzero["regularized"] and not nonzero["incumbent"]:
                            changed_only[horizon][name].append(error)
            if start_step <= step <= until_step and (step - start_step) % stride == 0:
                states = np.asarray(obs["balloon_states"], dtype=float)
                status = np.asarray(obs["balloon_status"]).reshape(-1)
                released = np.flatnonzero(
                    (status == 1) & np.all(np.isfinite(states), axis=1),
                )
                for horizon, steps in horizon_steps:
                    for index in released:
                        base = states[index, :2] + states[index, 3:5] * horizon
                        old = incumbent.correction(states, status, index, horizon)
                        new = candidate.correction(states, status, index, horizon)
                        predictions = {
                            "constant_velocity": base,
                            "incumbent": base + blend * old,
                            "regularized": base + blend * new,
                        }
                        nonzero = {
                            "constant_velocity": False,
                            "incumbent": bool(np.any(old)),
                            "regularized": bool(np.any(new)),
                        }
                        due[step + steps].append((int(index), horizon,
                                                   predictions, nonzero))
            if step == end_step:
                break
            obs, _, terminated, truncated, _ = env.step(no_launch)
            if terminated or truncated:
                raise RuntimeError("Official environment ended before forecast check")
    finally:
        env.close()

    return {
        "seed": int(seed),
        "source_sha256": {
            str(Path(module.__file__).as_posix()): hashlib.sha256(
                Path(module.__file__).read_bytes(),
            ).hexdigest()
            for module in (scenario4_wind_profile_agent,
                           scenario4_regularized_wind_agent, balloon_world)
        },
        "sampling": {"start_s": start, "until_s": until,
                     "interval_s": sample_interval,
                     "horizons_s": list(horizons), "blend": blend},
        "by_horizon": {
            str(h): {
                "methods": {
                    name: summarize(values, corrected[h][name])
                    for name, values in errors[h].items()
                },
                "paired_mean_squared_error_delta_m2": float(np.mean(
                    np.asarray(errors[h]["regularized"]) ** 2 -
                    np.asarray(errors[h]["incumbent"]) ** 2,
                )) if errors[h]["incumbent"] else None,
                "newly_supported_count": len(changed_only[h]["regularized"]),
                "newly_supported": {
                    name: summarize(values, len(values) if name == "regularized" else 0)
                    for name, values in changed_only[h].items()
                },
            } for h, _ in horizon_steps
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--start", type=float, default=24.0)
    parser.add_argument("--until", type=float, default=60.0)
    parser.add_argument("--interval", type=float, default=2.0)
    args = parser.parse_args()
    result = []
    for seed in args.seeds:
        print(f"Forecast validation on fresh official Scenario 4 seed {seed}",
              flush=True)
        result.append(evaluate_seed(seed, start=args.start, until=args.until,
                                    sample_interval=args.interval))
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(result, indent=2, allow_nan=False),
                               encoding="utf-8")
    print(f"Wrote {args.report}")


if __name__ == "__main__":
    main()
