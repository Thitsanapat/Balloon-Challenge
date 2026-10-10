"""Summarize offline shadow predictions; never used inside an agent.

Only evaluator-generated diagnostics are accepted. Official truth in these
reports is used for post-flight validation, not returned to a controller.
"""

import argparse
import json
from pathlib import Path
from statistics import mean


def _p90(values):
    ordered = sorted(values)
    if not ordered:
        return None
    return ordered[int(0.9 * (len(ordered) - 1))]


def _metrics(predictions, radius):
    if not predictions:
        return {"count": 0, "position_error_mean_m": None,
                "position_error_p90_m": None,
                "closest_error_mean_m": None,
                "closest_error_p90_m": None,
                "predicted_capture_count": 0,
                "false_predicted_capture_count": 0}
    position_errors = [float(item["rocket_position_error"])
                       for item in predictions]
    closest_errors = [abs(float(item["closest_distance_error"]))
                      for item in predictions]
    predicted_captures = [item for item in predictions
                          if float(item["predicted_closest_distance"]) <= radius]
    false_captures = [item for item in predicted_captures
                      if float(item["actual_closest_distance"]) > radius]
    return {"count": len(predictions),
            "position_error_mean_m": mean(position_errors),
            "position_error_p90_m": _p90(position_errors),
            "closest_error_mean_m": mean(closest_errors),
            "closest_error_p90_m": _p90(closest_errors),
            "predicted_capture_count": len(predicted_captures),
            "false_predicted_capture_count": len(false_captures)}


def summarize_reports(reports, *, capture_radius=1.5):
    """Aggregate held-plan and changed-plan forecasts without seed leakage."""
    if capture_radius <= 0:
        raise ValueError("capture_radius must be positive")
    horizons = set()
    forecasts = []
    seeds = []
    for report in reports:
        if (not isinstance(report, dict) or report.get("scenario") != 4 or
                not str(report.get("scope", "")).startswith(
                    "Evaluator-only fresh Scenario 4 shadow validation")):
            raise ValueError("Expected a Scenario 4 shadow diagnostic report")
        seeds.append(int(report["seed"]))
        horizons.update(float(value) for value in report["horizons"])
        forecasts.extend(item for anchor in report["anchors"]
                         for item in anchor.get("predictions", [])
                         if "rocket_position_error" in item)
    result = {"seeds": seeds, "reports": len(reports),
              "capture_radius_m": capture_radius, "horizons": {}}
    for horizon in sorted(horizons):
        group = [item for item in forecasts
                 if float(item["horizon"]) == horizon]
        fixed = [item for item in group
                 if item["plan_changes"] == 0 and item["target_changes"] == 0]
        result["horizons"][str(horizon)] = {
            "all": _metrics(group, capture_radius),
            "fixed_plan": _metrics(fixed, capture_radius),
            "changed_plan_count": len(group) - len(fixed),
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", type=Path, nargs="+")
    parser.add_argument("--capture-radius", type=float, default=1.5)
    args = parser.parse_args()
    reports = [json.loads(path.read_text(encoding="utf-8"))
               for path in args.reports]
    print(json.dumps(summarize_reports(
        reports, capture_radius=args.capture_radius), indent=2))


if __name__ == "__main__":
    main()
