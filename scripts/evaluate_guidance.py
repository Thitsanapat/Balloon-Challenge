"""Fresh official-environment evaluations with explicit seeds and JSON metrics.

This does not pack/upload leaderboard submissions or modify scenario YAML.
"""

import argparse
import copy
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

from BalloonPoppingGymEnv.envs.balloon_world import BalloonPoppingEnv
from BalloonPoppingGymEnv.evaluation.evaluate import (
    _load_agent_class, load_scenario_parameters,
)


def agent_source_fingerprints(main_path):
    root = Path(__file__).resolve().parents[1]
    agent_dir = root/'BalloonPoppingGymEnv/agents'
    paths = {Path(main_path).resolve()}
    for module in tuple(sys.modules.values()):
        filename = getattr(module,'__file__',None)
        if filename:
            path = Path(filename).resolve()
            if path.suffix=='.py' and path.is_relative_to(agent_dir):
                paths.add(path)
    return {str(path.relative_to(root) if path.is_relative_to(root) else path):hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(paths)}


def evaluate(config, seed):
    agent_sha256 = hashlib.sha256(Path(config['agent_module_path']).read_bytes()).hexdigest()
    parameters, given = load_scenario_parameters(config["scenario_number"])
    parameters = copy.deepcopy(parameters)
    parameters["scenario"]["random_seed"] = seed
    env = BalloonPoppingEnv(render_mode=None, parameters=parameters)
    cls = _load_agent_class(config["agent_module_path"], config["agent_class_name"])
    dependencies_before = agent_source_fingerprints(config['agent_module_path'])
    agent = cls(given, **config.get("agent_kwargs", {}))
    start = time.monotonic()
    observation, info = env.reset(seed=seed)
    popped = set()
    events = []
    closest = np.full(len(observation['balloon_states']), np.inf)
    maximum_altitude = float(given['environment']['elevation'])
    terminated = truncated = False
    telemetry = None
    if config.get('record_guidance_telemetry', False):
        from scripts.guidance_telemetry import GuidanceTelemetry
        telemetry = GuidanceTelemetry()
    while not (terminated or truncated):
        action = agent.get_action(observation)
        if telemetry is not None:
            telemetry.before_step(agent, observation, action)
        observation, reward, terminated, truncated, info = env.step(action)
        if telemetry is not None:
            telemetry.after_step(observation)
        sensors = np.asarray(observation['rocket_sensors'], dtype=float)
        if np.all(np.isfinite(sensors[6:9])):
            maximum_altitude = max(maximum_altitude, float(sensors[8]))
            distances = np.linalg.norm(np.asarray(observation['balloon_states'])[:, :3]-sensors[6:9], axis=1)
            eligible = np.asarray(observation['balloon_status']).reshape(-1)==1
            closest[eligible] = np.minimum(closest[eligible], distances[eligible])
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
        "agent_sha256": agent_sha256,
        "agent_file_unchanged_during_run": agent_sha256 == hashlib.sha256(Path(config['agent_module_path']).read_bytes()).hexdigest(),
        "agent_source_dependencies_sha256": dependencies_before,
        "agent_sources_unchanged_during_run": dependencies_before == agent_source_fingerprints(config['agent_module_path']),
        "score": int(info["popped_count"]),
        "final_time": float(observation["simulation_time"]),
        "terminated": bool(terminated), "truncated": bool(truncated),
        "wall_seconds": time.monotonic()-start, "pop_events": events,
        "target_events": getattr(agent, "target_events", []),
        "diagnostics": getattr(agent, "diagnostics", {}),
        "maximum_observed_altitude": maximum_altitude,
        "closest_observed_distances": [float(d) if np.isfinite(d) else None for d in closest],
        "route_events": getattr(agent, "route_events", []),
        "bridge_events": getattr(agent, "bridge_events", []),
        "coverage_events": getattr(agent, "coverage_events", []),
        "launch_comparisons": getattr(agent, "launch_comparisons", []),
        "guidance_telemetry": telemetry.finish(observation['simulation_time']) if telemetry else None,
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
    parser.add_argument("--telemetry", action="store_true", help="Record observation/reference error decomposition; never fed back to the agent")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8-sig"))
    if args.telemetry:
        config['record_guidance_telemetry'] = True
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
