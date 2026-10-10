"""Audit factual two-leg continuation in observation-only route archives.

This is an offline diagnostic, never an agent feature provider.  Unflown
second legs are *censored*, not failures: their outcomes cannot be inferred
from the planned route.  Each first decision/attempt is counted once so a
controller's frequent replanning does not manufacture extra training rows.
"""

import argparse
import json
from pathlib import Path

import numpy as np


def actuator_features(snapshot):
    """Reduce an observation/agent-state snapshot to bounded scalar features."""
    if not isinstance(snapshot, dict):
        return None
    q, desired = snapshot.get("estimated_quaternion"), snapshot.get("desired_axis")
    gyro, gimbal = snapshot.get("gyro"), snapshot.get("estimated_gimbal_output")
    if any(item is None for item in (q, desired, gyro, gimbal)):
        return None
    q, desired, gyro, gimbal = (np.asarray(item, dtype=float)
                                for item in (q, desired, gyro, gimbal))
    if (q.shape != (4,) or desired.shape != (3,) or gyro.shape != (3,) or
            gimbal.shape != (2,) or
            not np.all(np.isfinite(np.r_[q, desired, gyro, gimbal])) or
            np.linalg.norm(q) < 1e-8 or np.linalg.norm(desired) < 1e-8):
        return None
    w, x, y, z = q / np.linalg.norm(q)
    body_axis = np.array([2*(x*z+w*y), 2*(y*z-w*x),
                          1-2*(x*x+y*y)])
    desired = desired / np.linalg.norm(desired)
    angle = float(np.degrees(np.arccos(np.clip(np.dot(body_axis, desired), -1., 1.))))
    max_gimbal = snapshot.get("max_gimbal_angle")
    fraction = (float(np.max(np.abs(gimbal)))/float(max_gimbal)
                if max_gimbal is not None and float(max_gimbal) > 0 else None)
    return dict(axis_error_degrees=angle,
                gyro_norm=float(np.linalg.norm(gyro)),
                gimbal_fraction=fraction)


def attempt_examples(events, seed):
    """Return executable first-leg attempts with factual continuation labels."""
    starts, outcomes = {}, {}
    decision_routes = {}
    for event in events:
        kind = event.get("kind")
        if kind == "decision":
            attempt = event["attempt_id"]
            decision_routes.setdefault(attempt, []).append(event)
            starts.setdefault(attempt, event)
        elif kind == "outcome":
            if event["attempt_id"] in outcomes:
                raise ValueError("Duplicate attempt outcome")
            outcomes[event["attempt_id"]] = event
    rows = []
    ordered = sorted(starts, key=lambda key: (starts[key]["time"], key))
    for position, attempt in enumerate(ordered):
        decision = starts[attempt]
        if not decision.get("route_metadata_aligned"):
            continue
        prefixes = decision.get("route_prefixes") or []
        if not prefixes or len(prefixes[-1]["targets"]) < 2:
            continue
        route = prefixes[-1]["targets"]
        durations = prefixes[-1]["durations"]
        outcome = outcomes.get(attempt)
        if outcome is None or outcome["outcome"] not in ("hit", "miss"):
            continue
        first_hit = outcome["outcome"] == "hit"
        first_pair_stable = all(
            (row.get("route_metadata_aligned") and
             row.get("route_prefixes") and
             row["route_prefixes"][-1]["targets"][:2] == route[:2])
            for row in decision_routes[attempt]
        )
        second_hit = None
        censor_reason = "first_leg_missed" if not first_hit else "next_target_changed"
        if first_hit and first_pair_stable and position + 1 < len(ordered):
            next_attempt = ordered[position + 1]
            next_decision = starts[next_attempt]
            if (next_decision["target"] == route[1] and
                    next_decision["time"] >= outcome["time"] - 1e-8):
                next_outcome = outcomes.get(next_attempt)
                if next_outcome and next_outcome["outcome"] in ("hit", "miss"):
                    second_hit = next_outcome["outcome"] == "hit"
                    censor_reason = None
                else:
                    censor_reason = "second_leg_unresolved"
        elif first_hit and not first_pair_stable:
            censor_reason = "route_changed_during_first_leg"
        # The stored route prefix and end states are observation/planner data;
        # no simulator truth or inferred alternative outcome enters a feature.
        end_velocities = np.asarray(decision.get("ends_v", []), dtype=float)
        departure_turn = None
        if end_velocities.ndim == 2 and len(end_velocities) >= 2:
            first, second = end_velocities[:2]
            denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
            if denominator > 1e-9:
                departure_turn = float(np.clip(np.dot(first, second) / denominator,
                                               -1.0, 1.0))
        rows.append(dict(seed=int(seed), attempt_id=int(attempt),
                         time=float(decision["time"]),
                         first_target=int(route[0]), second_target=int(route[1]),
                         planned_route_length=len(route),
                         planned_first_seconds=float(durations[0]),
                         planned_second_seconds=float(durations[1]),
                         planned_departure_cosine=departure_turn,
                         actuator=actuator_features(decision.get("actuator_snapshot")),
                         first_hit=first_hit, second_hit=second_hit,
                         second_censor_reason=censor_reason))
    return rows


def summarize(rows):
    observed = [row for row in rows if row["second_hit"] is not None]
    return dict(fields=len({row["seed"] for row in rows}),
                planned_two_leg_attempts=len(rows),
                first_hits=sum(row["first_hit"] for row in rows),
                observed_second_legs=len(observed),
                second_hits=sum(row["second_hit"] for row in observed),
                second_censored=len(rows) - len(observed),
                censor_reasons={reason: sum(row["second_censor_reason"] == reason
                                            for row in rows)
                                for reason in sorted({row["second_censor_reason"]
                                                      for row in rows
                                                      if row["second_censor_reason"]})})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path,
                        help="Directory of per-seed subfolders containing decisions.jsonl")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        parser.error("Refusing to overwrite an existing report")
    rows = []
    for path in sorted(args.archive.glob("*/decisions.jsonl")):
        seed = int(path.parent.name)
        events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        rows.extend(attempt_examples(events, seed))
    if not rows:
        parser.error("No completed aligned two-leg attempts found")
    report = dict(scope="Factual executed attempts only; censored legs are not negatives",
                  summary=summarize(rows), rows=rows)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"]))


if __name__ == "__main__":
    main()
