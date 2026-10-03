"""Evaluator-only shadow-rollout validation on a fresh official Scenario 4 run.

The agent receives exactly the observation from ``env.reset/step``.  Official
``info['rocket_states']`` is read only *after* selecting the action and only by
this diagnostic runner, never by the agent or shadow predictor.  The output is
a local, credential-free experiment report, not a leaderboard submission.

Example (intentionally not run by the unit tests)::

    .venv/Scripts/python scripts/validate_scenario4_shadow.py --seed 0
"""

import argparse
import copy
from datetime import datetime, timezone
import json
from math import comb
from pathlib import Path

import numpy as np

from BalloonPoppingGymEnv.agents.scenario4_shadow_predictor import (
    _segment_closest,
    predict_first_leg,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (ROOT / "BalloonPoppingGymEnv/evaluation/configs/"
                  "scenario4_wind_profile_launch42.yaml")
DEFAULT_RESULTS = ROOT / "BalloonPoppingGymEnv/evaluation/results"


def window_curve(curve, full_duration, elapsed, horizon):
    """Re-express a normalized quintic over [elapsed, elapsed + horizon]."""
    curve = np.asarray(curve, dtype=float)
    full_duration = float(full_duration)
    elapsed = float(elapsed)
    horizon = float(horizon)
    if (curve.shape != (6, 3) or not np.all(np.isfinite(curve)) or
            not np.all(np.isfinite([full_duration, elapsed, horizon])) or
            full_duration <= 0 or elapsed < 0 or horizon <= 0 or
            elapsed + horizon > full_duration + 1e-8):
        raise ValueError("invalid quintic reference window")
    start = elapsed / full_duration
    scale = horizon / full_duration
    shifted = np.zeros_like(curve)
    for order in range(6):
        shifted[order] = sum(
            comb(power, order) * start**(power - order) * curve[power]
            for power in range(order, 6)
        ) * scale**order
    return shifted


def _plan_key(agent):
    plan = getattr(agent, "plan", None)
    if plan is None:
        return None
    # Numerical coefficients are included because a refresh could keep the
    # same target/deadline while changing the actual commanded reference.
    return (getattr(agent, "target_index", None),
            float(agent.plan_start), float(agent.plan_duration),
            np.asarray(plan, dtype=float).tobytes())


def _truth_position(info):
    state = np.asarray(info.get("rocket_states", []), dtype=float)
    if state.shape != (13,) or not np.all(np.isfinite(state[:3])):
        return None
    return state[:3].copy()


def _prediction_at_anchor(agent, observation, info, given, offset, horizons,
                          previous_outputs, uncertainty):
    """Use only agent estimates and observation for prediction inputs."""
    now = float(observation["simulation_time"])
    target = getattr(agent, "target_index", None)
    plan = getattr(agent, "plan", None)
    record = {"requested_post_launch_offset": float(offset),
              "anchor_time": now, "target_index": target,
              "predictions": []}
    if target is None or plan is None:
        record["skip_reason"] = "no_active_target_or_plan"
        return record, []
    target = int(target)
    status = np.asarray(observation["balloon_status"]).reshape(-1)
    states = np.asarray(observation["balloon_states"], dtype=float)
    if (not 0 <= target < len(status) or status[target] != 1 or
            not np.all(np.isfinite(states[target]))):
        record["skip_reason"] = "target_not_released_or_invalid"
        return record, []
    truth_start = _truth_position(info)
    if truth_start is None:
        record["skip_reason"] = "no_finite_offline_truth_at_anchor"
        return record, []
    state_names = ("filtered_position", "filtered_velocity", "quaternion",
                   "filtered_gyro", "disturbance")
    estimates = [getattr(agent, name, None) for name in state_names]
    if any(value is None or not np.all(np.isfinite(value)) for value in estimates):
        record["skip_reason"] = "no_finite_agent_state_estimate"
        return record, []
    elapsed = now - float(agent.plan_start)
    remaining = float(agent.plan_duration) - elapsed
    record["remaining_plan_seconds"] = remaining
    target_state = states[target].copy()
    record["observed_target_position"] = target_state[:3].tolist()
    record["observed_target_velocity"] = target_state[3:6].tolist()
    active = []
    for horizon in horizons:
        if horizon > remaining + 1e-8:
            record["predictions"].append({
                "horizon": horizon, "skip_reason": "plan_ends_before_horizon"})
            continue
        local_curve = window_curve(plan, agent.plan_duration, elapsed, horizon)
        result = predict_first_leg(
            given_parameters=given, curve=local_curve, duration=horizon,
            now=now, launch_time=float(agent.launch_time),
            position=estimates[0], velocity=estimates[1],
            quaternion=estimates[2], angular_velocity=estimates[3],
            gimbal_output=previous_outputs[0],
            throttle_output=previous_outputs[1],
            roll_output=previous_outputs[2],
            target_position=target_state[:3],
            target_velocity=target_state[3:6],
            disturbance=estimates[4],
            tracking_frequency=float(agent.tracking_frequency),
            attitude_frequency=float(agent.attitude_frequency),
            max_tilt_degrees=float(np.degrees(agent.max_tilt)),
            max_axis_rate=float(agent.max_axis_rate),
            **uncertainty,
        )
        prediction = {
            "horizon": float(horizon),
            "predicted_rocket_position": list(result.final_position),
            "predicted_closest_distance": result.closest_distance,
            "nominal_spline_closest_distance": result.nominal_closest_distance,
            "three_sigma_distance_lower": result.three_sigma_distance_lower,
            "three_sigma_distance_upper": result.three_sigma_distance_upper,
            "predicted_max_reference_error": result.max_reference_error,
            "predicted_max_attitude_error_degrees":
                result.max_attitude_error_degrees,
            "predicted_acceleration_saturated_steps":
                result.acceleration_saturated_steps,
            "predicted_gimbal_limited_steps": result.gimbal_limited_steps,
            "predicted_throttle_limited_steps": result.throttle_limited_steps,
            "predicted_burnout_steps": result.burnout_steps,
        }
        record["predictions"].append(prediction)
        active.append({
            "prediction": prediction, "deadline": now + horizon,
            "target": target, "last_time": now,
            "last_truth": truth_start.copy(),
            "last_balloon": target_state[:3].copy(),
            "closest_true_distance": float(np.linalg.norm(
                truth_start - target_state[:3])),
            "plan_key": _plan_key(agent), "last_seen_plan_key": _plan_key(agent),
            "plan_changes": 0, "last_seen_target": target,
            "target_changes": 0,
        })
    if not active:
        record["skip_reason"] = "no_horizon_fits_active_plan"
    return record, active


def _observe_plan_change(active, agent):
    """Count plan changes without altering the agent or its observations."""
    key = _plan_key(agent)
    target = getattr(agent, "target_index", None)
    for pending in active:
        if key != pending["last_seen_plan_key"]:
            pending["plan_changes"] += 1
            pending["last_seen_plan_key"] = key
        if target != pending["last_seen_target"]:
            pending["target_changes"] += 1
            pending["last_seen_target"] = target


def _update_truth_interval(pending, observation, info):
    """Compare against *later* official truth, interpolating exact horizon."""
    current_time = float(observation["simulation_time"])
    if current_time <= pending["last_time"]:
        return False
    rocket_end = _truth_position(info)
    if rocket_end is None:
        pending["prediction"]["skip_reason"] = "official_truth_became_invalid"
        return True
    balloon_end = np.asarray(observation["balloon_states"], dtype=float)[
        pending["target"], :3]
    fraction = min(1.0, (pending["deadline"] - pending["last_time"]) /
                   (current_time - pending["last_time"]))
    fraction = max(0.0, fraction)
    truth_at_cutoff = (pending["last_truth"] +
                       fraction * (rocket_end - pending["last_truth"]))
    balloon_at_cutoff = (pending["last_balloon"] +
                         fraction * (balloon_end - pending["last_balloon"]))
    distance, _ = _segment_closest(
        pending["last_truth"], truth_at_cutoff,
        pending["last_balloon"], balloon_at_cutoff)
    pending["closest_true_distance"] = min(
        pending["closest_true_distance"], distance)
    pending["last_time"] = min(current_time, pending["deadline"])
    pending["last_truth"] = truth_at_cutoff
    pending["last_balloon"] = balloon_at_cutoff
    if current_time + 1e-9 < pending["deadline"]:
        return False
    report = pending["prediction"]
    predicted = np.asarray(report["predicted_rocket_position"], dtype=float)
    report.update(
        actual_rocket_position=truth_at_cutoff.tolist(),
        actual_closest_distance=pending["closest_true_distance"],
        rocket_position_error=float(np.linalg.norm(predicted - truth_at_cutoff)),
        closest_distance_error=(report["predicted_closest_distance"] -
                                pending["closest_true_distance"]),
        plan_changes=pending["plan_changes"],
        target_changes=pending["target_changes"],
    )
    return True


def run_episode(env, agent, given, seed, anchor_offsets, horizons, uncertainty):
    """Run one full official episode when explicitly invoked by the CLI.

    ``info`` is never passed to ``agent.get_action`` or ``predict_first_leg``.
    The agent always receives the unmodified official observation object.
    """
    observation, info = env.reset(seed=seed)
    terminated = truncated = False
    actual_launch_time = None
    anchors = []
    pending = []
    sampled = set()
    while not (terminated or truncated):
        previous_outputs = (
            np.asarray(getattr(agent, "previous_tvc", np.zeros(2)),
                       dtype=float).copy(),
            float(getattr(agent, "previous_throttle", 1.0)),
            float(getattr(agent, "previous_roll", 0.0)),
        )
        # Crucially, the original observation is passed unchanged to the agent.
        action = agent.get_action(observation)
        if pending:
            _observe_plan_change(pending, agent)
        if actual_launch_time is not None:
            now = float(observation["simulation_time"])
            for offset in anchor_offsets:
                if offset in sampled or now + 1e-9 < actual_launch_time + offset:
                    continue
                sampled.add(offset)
                record, new_pending = _prediction_at_anchor(
                    agent, observation, info, given, offset, horizons,
                    previous_outputs, uncertainty)
                anchors.append(record)
                pending.extend(new_pending)
        was_launched = bool(env.rocket_launched)
        observation, reward, terminated, truncated, info = env.step(action)
        if not was_launched and env.rocket_launched:
            actual_launch_time = float(observation["simulation_time"])
        still_pending = []
        for forecast in pending:
            if not _update_truth_interval(forecast, observation, info):
                still_pending.append(forecast)
        pending = still_pending
    for forecast in pending:
        forecast["prediction"]["skip_reason"] = "episode_ended_before_horizon"
    for offset in anchor_offsets:
        if offset not in sampled:
            anchors.append({"requested_post_launch_offset": offset,
                            "skip_reason": "episode_ended_before_anchor"})
    return {
        "scope": ("Evaluator-only fresh Scenario 4 shadow validation. Official "
                  "info truth is diagnostic-only and never supplied to the agent."),
        "scenario": 4,
        "seed": int(seed),
        "launch_time": actual_launch_time,
        "score": int(info.get("popped_count", 0)),
        "final_time": float(observation["simulation_time"]),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "anchor_offsets": list(anchor_offsets),
        "horizons": list(horizons),
        "uncertainty_inputs": dict(uncertainty),
        "limitations": ("The shadow model omits aerodynamic forces/moments and "
                        "future wind/balloon acceleration. Predictions hold one "
                        "reference spline fixed, while the real agent may replan; "
                        "plan_changes is reported for each comparison. Three-sigma "
                        "bounds are pointwise heuristics, not capture probabilities."),
        "anchors": anchors,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--anchor-offsets", nargs="+", type=float,
                        default=[0.5, 2.0, 5.0, 8.0])
    parser.add_argument("--horizons", nargs="+", type=float,
                        default=[0.25, 1.0, 2.0, 3.0])
    parser.add_argument("--state-sigma", type=float, default=0.0)
    parser.add_argument("--target-position-sigma", type=float, default=0.0)
    parser.add_argument("--target-velocity-sigma", type=float, default=0.0)
    parser.add_argument("--model-acceleration-sigma", type=float, default=0.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if (not np.all(np.isfinite(args.anchor_offsets + args.horizons)) or
            min(args.anchor_offsets) < 0 or min(args.horizons) <= 0 or
            len(set(args.anchor_offsets)) != len(args.anchor_offsets) or
            len(set(args.horizons)) != len(args.horizons)):
        parser.error("anchors must be unique and nonnegative; horizons positive")
    uncertainty = dict(
        initial_position_sigma=args.state_sigma,
        target_position_sigma=args.target_position_sigma,
        target_velocity_sigma=args.target_velocity_sigma,
        model_acceleration_sigma=args.model_acceleration_sigma,
    )
    if any(not np.isfinite(value) or value < 0 for value in uncertainty.values()):
        parser.error("uncertainty sigmas must be finite and nonnegative")

    # Delay heavyweight environment imports until someone explicitly runs the
    # CLI; importing the pure helpers in unit tests never starts a simulation.
    import yaml
    from BalloonPoppingGymEnv.envs.balloon_world import BalloonPoppingEnv
    from BalloonPoppingGymEnv.evaluation.evaluate import (
        _load_agent_class, load_scenario_parameters,
    )

    config = yaml.safe_load(args.config.read_text(encoding="utf-8-sig"))
    if config.get("scenario_number") != 4:
        parser.error("the validation harness only accepts Scenario 4 configs")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = (args.output if args.output is not None else
              DEFAULT_RESULTS / f"s4_shadow_validation_seed{args.seed}_{timestamp}.json")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing report: {output}")
    parameters, given = load_scenario_parameters(4)
    parameters = copy.deepcopy(parameters)
    parameters["scenario"]["random_seed"] = args.seed
    module_path = Path(config["agent_module_path"])
    if not module_path.is_absolute():
        module_path = ROOT / module_path
    agent_class = _load_agent_class(str(module_path),
                                    config["agent_class_name"])
    agent = agent_class(given, **config.get("agent_kwargs", {}))
    env = BalloonPoppingEnv(render_mode=None, parameters=parameters)
    try:
        report = run_episode(env, agent, given, args.seed,
                             sorted(args.anchor_offsets), sorted(args.horizons),
                             uncertainty)
    finally:
        env.close()
    report["agent_class_name"] = config["agent_class_name"]
    report["config_path"] = str(args.config)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as destination:
        destination.write(json.dumps(report, indent=2, allow_nan=False))
    print(f"score={report['score']} anchors={len(report['anchors'])} report={output}")


if __name__ == "__main__":
    main()
