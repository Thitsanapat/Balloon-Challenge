"""Fresh official-environment evaluations with explicit seeds and JSON metrics.

This does not pack/upload leaderboard submissions or modify scenario YAML.
"""

import argparse
import copy
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import yaml

from BalloonPoppingGymEnv.envs.balloon_world import BalloonPoppingEnv
from BalloonPoppingGymEnv.evaluation.evaluate import (
    _load_agent_class, load_scenario_parameters,
)


def evaluate(config, seed):
    parameters, given = load_scenario_parameters(config["scenario_number"])
    parameters = copy.deepcopy(parameters)
    parameters["scenario"]["random_seed"] = seed
    env = BalloonPoppingEnv(render_mode=None, parameters=parameters)
    cls = _load_agent_class(config["agent_module_path"], config["agent_class_name"])
    agent = cls(given, **config.get("agent_kwargs", {}))
    start = time.monotonic()
    observation, info = env.reset(seed=seed)
    popped = set()
    events = []
    terminated = truncated = False
    while not (terminated or truncated):
        observation, reward, terminated, truncated, info = env.step(agent.get_action(observation))
        if reward:
            for index in np.flatnonzero(np.asarray(observation["balloon_status"]).reshape(-1) == 2):
                index = int(index)
                if index not in popped:
                    popped.add(index)
                    events.append([index, float(observation["simulation_time"])])
            print(f"seed={seed} score={info['popped_count']} time={observation['simulation_time']:.2f}", flush=True)
    result = {
        "evaluation": "fresh official BalloonPoppingEnv; no cached field substitution",
        "scenario": config["scenario_number"], "seed": seed, "config": config,
        "agent_sha256": hashlib.sha256(Path(config["agent_module_path"]).read_bytes()).hexdigest(),
        "score": int(info["popped_count"]),
        "final_time": float(observation["simulation_time"]),
        "terminated": bool(terminated), "truncated": bool(truncated),
        "wall_seconds": time.monotonic()-start, "pop_events": events,
        "target_events": getattr(agent, "target_events", []),
        "diagnostics": getattr(agent, "diagnostics", {}),
    }
    env.close()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument(
        "--scenario",
        type=int,
        choices=(0, 1, 2, 3),
        help="Override the config's scenario number for cross-scenario validation",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="Override an agent argument, recorded in the report")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8-sig"))
    if args.scenario is not None:
        config["scenario_number"] = args.scenario
    for assignment in args.set:
        key, value = assignment.split("=", 1)
        config.setdefault("agent_kwargs", {})[key] = yaml.safe_load(value)
    results = []
    for seed in args.seeds:
        print(f"Starting fresh evaluation: {config['agent_name']}, seed={seed}", flush=True)
        result = evaluate(config, seed)
        results.append(result)
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(results, indent=2, allow_nan=False), encoding="utf-8")
        print(f"COMPLETE seed={seed} score={result['score']} report={args.report}", flush=True)


if __name__ == "__main__":
    main()
