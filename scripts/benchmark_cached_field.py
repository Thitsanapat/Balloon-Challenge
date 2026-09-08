"""Run an agent against balloon paths cached in an earlier deterministic run."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import yaml

from BalloonPoppingGymEnv.envs.balloon_world import BalloonPoppingEnv
from BalloonPoppingGymEnv.evaluation.evaluate import load_scenario_parameters
from BalloonPoppingGymEnv.agents.multi_target_agent import _rotate_body_to_world


def load_agent(path, class_name):
    spec = importlib.util.spec_from_file_location("cached_benchmark_agent", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, class_name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("trajectory", type=Path)
    parser.add_argument("--end-time", type=float)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--report", type=Path, help="Save reproducible metrics, not a submission")
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8-sig"))
    for assignment in args.set:
        key, value = assignment.split("=", 1)
        config.setdefault("agent_kwargs", {})[key] = yaml.safe_load(value)
    scenario, given = load_scenario_parameters(config["scenario_number"])
    dt = float(scenario["simulation"]["time_step"])
    cached_release_steps = None
    if args.trajectory.suffix.lower() == ".npz":
        with np.load(args.trajectory) as cache:
            cached_flights = np.asarray(cache["balloon_flights"])
            cached_release_steps = np.asarray(cache["release_steps"], dtype=int)
            if int(cache["scenario"]) != config["scenario_number"]:
                raise ValueError("Cached field scenario does not match configuration")
            if int(cache["seed"]) != scenario["scenario"]["random_seed"]:
                raise ValueError("Cached field seed does not match configured seed")
            if not np.isclose(float(cache["time_step"]), dt):
                raise ValueError("Cached field time step does not match scenario")
        if args.end_time is not None:
            samples = min(cached_flights.shape[2], int(args.end_time / dt) + 1)
            cached_flights = cached_flights[:, :, :samples]
        field_end_time = (cached_flights.shape[2] - 1) * dt
    else:
        records = json.loads(args.trajectory.read_text(encoding="utf-8"))
        if args.end_time is not None:
            records = [record for record in records if record["time"] <= args.end_time]
        recorded = np.asarray([record["balloon_states"] for record in records])
        if not records:
            raise ValueError("The requested cached-field interval contains no samples")
        times = np.asarray([record["time"] for record in records])
        if not np.allclose(times, dt*np.arange(1, len(records)+1), atol=1e-8, rtol=0):
            raise ValueError("Cached field must contain consecutive timesteps starting at dt")
        if recorded.shape[1:] != (scenario["balloon"]["num"], 6):
            raise ValueError("Cached balloon count/state dimension does not match the scenario")
        cached_flights = np.transpose(
            np.concatenate((recorded[:1], recorded), axis=0), (1, 2, 0)
        )
        field_end_time = records[-1]["time"]

    if cached_flights.shape[:2] != (scenario["balloon"]["num"], 6):
        raise ValueError("Cached balloon count/state dimension does not match scenario")

    env = BalloonPoppingEnv(render_mode=None, parameters=scenario)
    env._BalloonPoppingEnv__generate_balloon_flights = lambda: setattr(
        env, "_balloon_flights", cached_flights
    )
    observation, info = env.reset(seed=scenario["scenario"]["random_seed"])
    if cached_release_steps is not None and not np.array_equal(
        cached_release_steps, env._balloon_release_at_step
    ):
        raise ValueError("Cached release schedule does not match configured seed")
    # Harness validation only: never pass the cache or release schedule to agent.
    if args.trajectory.suffix.lower() != ".npz" and config["scenario_number"] != 0:
        statuses = np.asarray([record["balloon_status"] for record in records])
        released = np.arange(1, len(records)+1)[:, None] >= env._balloon_release_at_step
        if not np.array_equal(statuses > 0, released):
            raise ValueError("Cached release schedule does not match the configured seed")
    agent_class = load_agent(config["agent_module_path"], config["agent_class_name"])
    agent = agent_class(given, **config.get("agent_kwargs", {}))

    popped = []
    closest = np.full(scenario["balloon"]["num"], np.inf)
    closest_time = np.zeros(scenario["balloon"]["num"])
    closest_offset = np.zeros((scenario["balloon"]["num"], 3))
    closest_actual_axis = np.zeros((scenario["balloon"]["num"], 3))
    closest_desired_axis = np.zeros((scenario["balloon"]["num"], 3))
    terminated = truncated = False
    while not (terminated or truncated):
        action = agent.get_action(observation)
        observation, reward, terminated, truncated, info = env.step(action)
        rocket_position = np.asarray(info["rocket_states"][:3], dtype=float)
        if np.all(np.isfinite(rocket_position)):
            balloon_positions = np.asarray(observation["balloon_states"])[:, :3]
            released = np.asarray(observation["balloon_status"]).reshape(-1) == 1
            ranges = np.linalg.norm(balloon_positions - rocket_position, axis=1)
            improved = released & (ranges < closest)
            closest[improved] = ranges[improved]
            closest_time[improved] = float(observation["simulation_time"])
            closest_offset[improved] = (
                balloon_positions[improved] - rocket_position
            )
            quaternion = np.asarray(info["rocket_states"][6:10], dtype=float)
            if np.all(np.isfinite(quaternion)):
                closest_actual_axis[improved] = _rotate_body_to_world(
                    quaternion, np.array([0.0, 0.0, 1.0])
                )
            closest_desired_axis[improved] = getattr(
                agent, "last_desired_axis", np.array([0.0, 0.0, 1.0])
            )
        if reward:
            status = np.asarray(observation["balloon_status"]).reshape(-1)
            known = {index for index, _ in popped}
            for index in np.flatnonzero(status == 2):
                if int(index) not in known:
                    popped.append((int(index), float(observation["simulation_time"])))

    rocket = np.asarray(info["rocket_states"], dtype=float)
    print(
        f"score={info['popped_count']} final_time={observation['simulation_time']:.2f} "
        f"position={np.round(rocket[:3], 2)} velocity={np.round(rocket[3:6], 2)}"
    )
    print(" ".join(f"#{index}@{time:.2f}" for index, time in popped))
    order = np.argsort(closest)[:8]
    print(
        "closest "
        + " ".join(
            f"#{index}:{closest[index]:.2f}m@{closest_time[index]:.2f}"
            f"{np.round(closest_offset[index], 1)}"
            f" a={np.round(closest_actual_axis[index], 1)}"
            f" d={np.round(closest_desired_axis[index], 1)}"
            for index in order
            if np.isfinite(closest[index])
        )
    )
    if hasattr(agent, "target_events"):
        print(
            "targets "
            + " ".join(f"#{index}@{time:.2f}" for time, index in agent.target_events)
        )
    diagnostics = getattr(agent, "diagnostics", {})
    print("diagnostics", diagnostics)
    if args.report is not None:
        report = {
            "evaluation": "cached-field development benchmark; not official submission",
            "config": config,
            "agent_sha256": hashlib.sha256(Path(config["agent_module_path"]).read_bytes()).hexdigest(),
            "seed": scenario["scenario"]["random_seed"],
            "field_source": str(args.trajectory),
            "field_end_time": field_end_time,
            "score": int(info["popped_count"]),
            "final_time": float(observation["simulation_time"]),
            "terminated": bool(terminated), "truncated": bool(truncated),
            "pop_events": popped,
            "target_events": getattr(agent, "target_events", []),
            "diagnostics": diagnostics,
        }
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()
