"""Observation-only diagnostic gate for a Scenario 4 first-target ZEM idea.

This runner does *not* apply ZEM controls.  It watches a normal agent's first
selected released target, probes static 3-second ZEM feasibility using its
sensor-derived navigation estimate and public vehicle envelope, then labels
the event from later balloon status and observed proximity.  It never reads
``info`` or simulator internals and never passes diagnostic data to the agent.

The gate is optimistic: it ignores actuator lag, future wind/target
acceleration, aerodynamic moments, and navigation uncertainty.  Passing it is
necessary evidence for trying a policy experiment, not evidence of a hit.
"""

import argparse
import copy
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from BalloonPoppingGymEnv.agents.scenario4_zem_terminal_agent import (
    zero_effort_miss_acceleration,
)
from BalloonPoppingGymEnv.agents.submission_sqp_v2 import (
    G,
    allocate_acceleration,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (ROOT / "BalloonPoppingGymEnv/evaluation/configs/"
                  "scenario4_wind_profile_launch42.yaml")
DEFAULT_RESULTS = ROOT / "BalloonPoppingGymEnv/evaluation/results"


def _finite_vector(value, length, name):
    array = np.asarray(value, dtype=float)
    if array.shape != (length,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite {length}-vector")
    return array


def _observed_body_axis(quaternion):
    quaternion = _finite_vector(quaternion, 4, "quaternion")
    norm = float(np.linalg.norm(quaternion))
    if norm <= 0:
        raise ValueError("zero attitude quaternion")
    w, x, y, z = quaternion / norm
    return np.array([2 * (x * z + w * y),
                     2 * (y * z - w * x),
                     1 - 2 * (x * x + y * y)])


def zem_feasibility_probe(relative_position, relative_velocity, disturbance,
                          time_to_go, available, max_tilt, body_axis,
                          max_axis_rate, force_tolerance=0.1):
    """Static public-envelope test at one observation; no simulator state."""
    relative_position = _finite_vector(relative_position, 3,
                                       "relative_position")
    relative_velocity = _finite_vector(relative_velocity, 3,
                                       "relative_velocity")
    disturbance = _finite_vector(disturbance, 3, "disturbance")
    body_axis = _finite_vector(body_axis, 3, "body_axis")
    values = (time_to_go, available, max_tilt, max_axis_rate,
              force_tolerance)
    if (not np.all(np.isfinite(values)) or time_to_go <= 0 or
            available < 0 or not 0 <= max_tilt < np.pi / 2 or
            max_axis_rate <= 0 or force_tolerance < 0 or
            np.linalg.norm(body_axis) <= 0):
        raise ValueError("invalid ZEM feasibility settings")
    body_axis = body_axis / np.linalg.norm(body_axis)
    zem = relative_position + relative_velocity * time_to_go
    acceleration = zero_effort_miss_acceleration(
        relative_position, relative_velocity, time_to_go)
    required = acceleration - G - disturbance
    allocated = allocate_acceleration(required, available, max_tilt)
    allocation_loss = float(np.linalg.norm(allocated - required))
    required_norm = float(np.linalg.norm(required))
    if required_norm > 1e-9:
        desired_axis = required / required_norm
        turn_angle = float(np.arccos(np.clip(
            np.dot(body_axis, desired_axis), -1.0, 1.0)))
    else:
        turn_angle = 0.0
    turn_headroom = max_axis_rate * time_to_go - turn_angle
    return {
        "time_to_go": float(time_to_go),
        "zero_effort_miss_m": float(np.linalg.norm(zem)),
        "required_force_acceleration": required.tolist(),
        "required_force_magnitude": required_norm,
        "available_force_acceleration": float(available),
        "thrust_headroom": float(available - required_norm),
        "allocation_loss": allocation_loss,
        "turn_headroom_radians": float(turn_headroom),
        "feasible": bool(all((available > 0,
                              required_norm >= 0.5,
                              allocation_loss <= force_tolerance,
                              turn_headroom >= 0))),
    }


def _active_deadline(agent, target):
    route = getattr(agent, "route", [])
    deadlines = getattr(agent, "deadlines", [])
    if route and deadlines and route[0] == target:
        return float(deadlines[0])
    if getattr(agent, "target_index", None) == target and agent.plan is not None:
        return float(agent.plan_start + agent.plan_duration)
    return None


class FirstTargetGate:
    """Observe a flight; produce at most one first-target feasibility probe."""

    def __init__(self, probe_time=3.0, probe_tolerance=0.25,
                 post_deadline_grace=1.0):
        values = (probe_time, probe_tolerance, post_deadline_grace)
        if (not np.all(np.isfinite(values)) or probe_time <= 0 or
                probe_tolerance < 0 or post_deadline_grace < 0):
            raise ValueError("invalid first-target gate timing")
        self.probe_time = float(probe_time)
        self.probe_tolerance = float(probe_tolerance)
        self.post_deadline_grace = float(post_deadline_grace)
        self.first_target = None
        self.first_selected_time = None
        self.first_deadline = None
        self.probe = None
        self.popped_time = None
        self.minimum_observed_distance = float("inf")
        self.last_time = None

    def observe(self, observation, agent):
        """Call after agent.get_action, using only observation + agent state."""
        now = float(observation["simulation_time"])
        status = np.asarray(observation["balloon_status"]).reshape(-1)
        states = np.asarray(observation["balloon_states"], dtype=float)
        if self.last_time is not None and now <= self.last_time:
            return
        self.last_time = now
        if self.first_target is None:
            candidate = getattr(agent, "target_index", None)
            if (candidate is not None and 0 <= int(candidate) < len(status) and
                    status[int(candidate)] == 1):
                self.first_target = int(candidate)
                self.first_selected_time = now
                self.first_deadline = _active_deadline(agent, self.first_target)
        if self.first_target is None:
            return
        target = self.first_target
        if status[target] == 2 and self.popped_time is None:
            self.popped_time = now
        if (status[target] != 1 or
                not np.all(np.isfinite(states[target]))):
            return
        estimated_position = getattr(agent, "filtered_position", None)
        estimated_velocity = getattr(agent, "filtered_velocity", None)
        if (estimated_position is None or estimated_velocity is None or
                not np.all(np.isfinite(estimated_position)) or
                not np.all(np.isfinite(estimated_velocity))):
            return
        estimated_position = _finite_vector(estimated_position, 3,
                                            "estimated_position")
        estimated_velocity = _finite_vector(estimated_velocity, 3,
                                            "estimated_velocity")
        relative_position = states[target, :3] - estimated_position
        if (self.first_deadline is None or
                now <= self.first_deadline + self.post_deadline_grace):
            self.minimum_observed_distance = min(
                self.minimum_observed_distance,
                float(np.linalg.norm(relative_position)))
        if self.probe is not None or getattr(agent, "target_index", None) != target:
            return
        deadline = _active_deadline(agent, target)
        if deadline is None:
            return
        time_to_go = deadline - now
        if abs(time_to_go - self.probe_time) > self.probe_tolerance:
            return
        relative_velocity = states[target, 3:6] - estimated_velocity
        body_axis = _observed_body_axis(agent.quaternion)
        details = zem_feasibility_probe(
            relative_position, relative_velocity, agent.disturbance,
            time_to_go, agent.available_acceleration(now),
            agent.max_tilt, body_axis, agent.max_axis_rate)
        self.probe = {"time": now, "target_index": target,
                      "deadline": deadline, **details}

    def observe_terminal_status(self, observation):
        """Capture a pop on the final environment step without using info."""
        if self.first_target is None or self.popped_time is not None:
            return
        status = np.asarray(observation["balloon_status"]).reshape(-1)
        if status[self.first_target] == 2:
            self.popped_time = float(observation["simulation_time"])

    def finish(self):
        if self.first_target is None:
            return {"skip_reason": "no_released_first_target"}
        deadline = (self.probe["deadline"] if self.probe is not None else
                    self.first_deadline)
        popped_by_deadline = bool(
            self.popped_time is not None and deadline is not None and
            self.popped_time <= deadline + self.post_deadline_grace)
        return {
            "first_target_index": self.first_target,
            "first_selected_time": self.first_selected_time,
            "first_deadline": self.first_deadline,
            "probe": self.probe,
            "popped_time": self.popped_time,
            "popped_by_deadline": popped_by_deadline,
            "minimum_observed_distance": (
                self.minimum_observed_distance
                if np.isfinite(self.minimum_observed_distance) else None),
            "skip_reason": (None if self.probe is not None else
                            "no_observation_near_three_seconds_to_deadline"),
        }


def summarize_gate(events, miss_distance=5.0, minimum_events=20,
                   minimum_per_class=5, minimum_miss_recall=0.8,
                   maximum_hit_false_rejection=0.2):
    """Decision threshold for spending a subsequent full policy score run."""
    events = list(events)
    probed = [event for event in events if event.get("probe") is not None]
    hits = [event for event in probed if event.get("popped_by_deadline")]
    large_misses = [
        event for event in probed
        if not event.get("popped_by_deadline") and
        event.get("minimum_observed_distance") is not None and
        event["minimum_observed_distance"] > miss_distance
    ]
    rejected_misses = sum(not event["probe"]["feasible"]
                          for event in large_misses)
    rejected_hits = sum(not event["probe"]["feasible"] for event in hits)
    recall = rejected_misses / len(large_misses) if large_misses else None
    false_rejection = rejected_hits / len(hits) if hits else None
    enough = (len(probed) >= minimum_events and
              len(hits) >= minimum_per_class and
              len(large_misses) >= minimum_per_class)
    decision = ("insufficient_data" if not enough else
                "pass" if recall >= minimum_miss_recall and
                false_rejection <= maximum_hit_false_rejection else "fail")
    return {
        "probed_first_targets": len(probed),
        "timely_first_target_hits": len(hits),
        "large_observed_misses": len(large_misses),
        "large_miss_recall": recall,
        "hit_false_rejection_rate": false_rejection,
        "decision": decision,
        "threshold": {
            "minimum_probed_events": minimum_events,
            "minimum_hits_and_large_misses_each": minimum_per_class,
            "large_miss_distance_m": miss_distance,
            "minimum_large_miss_recall": minimum_miss_recall,
            "maximum_hit_false_rejection": maximum_hit_false_rejection,
        },
    }


def run_gate_episode(env, agent, seed):
    """No official ``info`` is read, retained, or passed to the agent."""
    observation, _ = env.reset(seed=seed)
    gate = FirstTargetGate()
    terminated = truncated = False
    while not (terminated or truncated):
        action = agent.get_action(observation)
        gate.observe(observation, agent)
        observation, _, terminated, truncated, _ = env.step(action)
    gate.observe_terminal_status(observation)
    event = gate.finish()
    event["score_from_observed_pop_status"] = int(np.sum(
        np.asarray(observation["balloon_status"]).reshape(-1) == 2))
    event["seed"] = int(seed)
    return event


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    # Imports here ensure unit tests never reset an official environment.
    import yaml
    from BalloonPoppingGymEnv.envs.balloon_world import BalloonPoppingEnv
    from BalloonPoppingGymEnv.evaluation.evaluate import (
        _load_agent_class, load_scenario_parameters,
    )

    config = yaml.safe_load(args.config.read_text(encoding="utf-8-sig"))
    if config.get("scenario_number") != 4 or config.get("leaderboard_submission"):
        parser.error("an offline Scenario 4 config is required")
    module_path = Path(config["agent_module_path"])
    if not module_path.is_absolute():
        module_path = ROOT / module_path
    agent_class = _load_agent_class(str(module_path),
                                    config["agent_class_name"])
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = (args.output if args.output is not None else
              DEFAULT_RESULTS / f"s4_zem_gate_{timestamp}.json")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing report: {output}")
    events = []
    for seed in args.seeds:
        parameters, given = load_scenario_parameters(4)
        parameters = copy.deepcopy(parameters)
        parameters["scenario"]["random_seed"] = seed
        agent = agent_class(given, **config.get("agent_kwargs", {}))
        env = BalloonPoppingEnv(render_mode=None, parameters=parameters)
        try:
            event = run_gate_episode(env, agent, seed)
        finally:
            env.close()
        events.append(event)
        print(f"seed={seed} observed_score={event['score_from_observed_pop_status']}",
              flush=True)
    report = {
        "scope": "Observation-only first-target ZEM feasibility diagnostic; no hidden info used",
        "scenario": 4,
        "config_path": str(args.config),
        "agent_class_name": config["agent_class_name"],
        "events": events,
        "summary": summarize_gate(events),
        "limitations": ("Static acceleration/turn gate is optimistic, and observed "
                        "proximity has GNSS error. Pop labels come only from "
                        "balloon_status. Passing the gate does not validate "
                        "the ZEM flight controller."),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as destination:
        destination.write(json.dumps(report, indent=2, allow_nan=False))
    print(f"gate={report['summary']['decision']} report={output}")


if __name__ == "__main__":
    main()
