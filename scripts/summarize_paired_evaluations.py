"""Compare saved fresh-evaluation reports on the same external seeds.

This is read-only analysis. It never runs an agent or changes an evaluation.
Reports should be produced by scripts/evaluate_guidance.py.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path


def _fingerprint(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_group(paths, label):
    rows = {}
    signature = None
    for path in paths:
        report = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        if isinstance(report, dict):
            report = [report]
        if not isinstance(report, list) or not report:
            raise ValueError(f"{path}: expected a nonempty evaluation report")
        for item in report:
            if not isinstance(item, dict):
                raise ValueError(f"{path}: expected evaluation objects")
            seed = item.get("seed")
            if type(seed) is not int or seed < 0:
                raise ValueError(f"{path}: invalid seed {seed!r}")
            if seed in rows:
                raise ValueError(f"{label}: duplicate seed {seed}")
            score = item.get("score")
            wall = item.get("wall_seconds")
            if type(score) is not int or score < 0:
                raise ValueError(f"{path}: invalid score for seed {seed}")
            if isinstance(wall, bool) or not isinstance(wall, (int, float)) or not math.isfinite(wall) or wall < 0:
                raise ValueError(f"{path}: invalid wall_seconds for seed {seed}")
            if type(item.get("terminated")) is not bool or type(item.get("truncated")) is not bool:
                raise ValueError(f"{path}: missing termination status for seed {seed}")
            source = item.get("agent_sha256")
            config = item.get("config")
            scenario = item.get("scenario")
            dependencies = item.get("agent_source_dependencies_sha256")
            if not isinstance(source, str) or not source or not isinstance(config, dict) or type(scenario) is not int:
                raise ValueError(f"{path}: missing source, config, or scenario for seed {seed}")
            current_signature = (scenario, source, _fingerprint(config), _fingerprint(dependencies))
            if signature is None:
                signature = current_signature
            elif current_signature != signature:
                raise ValueError(f"{label}: scenario, config, or source changed across reports")
            rows[seed] = item
    if not rows:
        raise ValueError(f"{label}: no reports supplied")
    return rows, signature


def summarize(baseline_paths, candidate_paths):
    """Return a paired summary; incomplete runs cannot contribute a score delta."""
    baseline, base_signature = _load_group(baseline_paths, "baseline")
    candidate, candidate_signature = _load_group(candidate_paths, "candidate")
    if base_signature[0] != candidate_signature[0]:
        raise ValueError("baseline and candidate use different scenarios")
    if baseline.keys() != candidate.keys():
        missing_baseline = sorted(candidate.keys() - baseline.keys())
        missing_candidate = sorted(baseline.keys() - candidate.keys())
        raise ValueError(f"seed sets differ: missing baseline {missing_baseline}; missing candidate {missing_candidate}")

    pairs = []
    deltas = []
    old_scores = []
    new_scores = []
    for seed in sorted(baseline):
        old, new = baseline[seed], candidate[seed]
        old_sources_ok = (old.get("agent_file_unchanged_during_run") is True
                          and old.get("agent_sources_unchanged_during_run") is True)
        new_sources_ok = (new.get("agent_file_unchanged_during_run") is True
                          and new.get("agent_sources_unchanged_during_run") is True)
        old_ok = old["terminated"] and not old["truncated"] and old_sources_ok
        new_ok = new["terminated"] and not new["truncated"] and new_sources_ok
        delta = new["score"] - old["score"] if old_ok and new_ok else None
        if delta is not None:
            deltas.append(delta)
            old_scores.append(old["score"])
            new_scores.append(new["score"])
        pairs.append({
            "seed": seed,
            "baseline_score": old["score"], "candidate_score": new["score"],
            "score_delta": delta,
            "baseline_wall_seconds": old["wall_seconds"],
            "candidate_wall_seconds": new["wall_seconds"],
            "baseline_reset_wall_seconds": old.get("reset_wall_seconds"),
            "candidate_reset_wall_seconds": new.get("reset_wall_seconds"),
            "baseline_post_reset_wall_seconds": old.get("post_reset_wall_seconds"),
            "candidate_post_reset_wall_seconds": new.get("post_reset_wall_seconds"),
            "baseline_terminated": old["terminated"],
            "candidate_terminated": new["terminated"],
            "baseline_truncated": old["truncated"],
            "candidate_truncated": new["truncated"],
            "baseline_sources_unchanged": old_sources_ok,
            "candidate_sources_unchanged": new_sources_ok,
        })
    count = len(deltas)
    all_complete = count == len(pairs)
    return {
        "scenario": base_signature[0],
        "baseline_source_sha256": base_signature[1],
        "candidate_source_sha256": candidate_signature[1],
        "baseline_config_sha256": base_signature[2],
        "candidate_config_sha256": candidate_signature[2],
        "baseline_dependencies_fingerprint": base_signature[3],
        "candidate_dependencies_fingerprint": candidate_signature[3],
        "seed_count": len(pairs), "complete_pair_count": count,
        "all_pairs_complete": all_complete,
        "mean_baseline_wall_seconds": sum(row["baseline_wall_seconds"] for row in pairs) / len(pairs),
        "mean_candidate_wall_seconds": sum(row["candidate_wall_seconds"] for row in pairs) / len(pairs),
        "mean_baseline_score": sum(old_scores) / count if all_complete else None,
        "mean_candidate_score": sum(new_scores) / count if all_complete else None,
        "mean_score_delta": sum(deltas) / count if all_complete else None,
        "total_score_delta": sum(deltas) if all_complete else None,
        "wins": sum(delta > 0 for delta in deltas),
        "ties": sum(delta == 0 for delta in deltas),
        "losses": sum(delta < 0 for delta in deltas),
        "pairs": pairs,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", nargs="+", required=True, type=Path)
    parser.add_argument("--candidate", nargs="+", required=True, type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(summarize(args.baseline, args.candidate), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
