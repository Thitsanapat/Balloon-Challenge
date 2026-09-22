"""Run an evaluation and create a leaderboard submission without saving secrets.

The credentials file is read at runtime.  Team credentials are kept out of the
tracked YAML configuration, while the official evaluator and submission packer
still receive the exact fields they require.
"""

import argparse
import hashlib
import json
import os
import re
from pathlib import Path

import yaml

from BalloonPoppingGymEnv.console_logging import configure_console_logging
from BalloonPoppingGymEnv.evaluation.evaluate import (
    _load_agent_class,
    evaluate_scenario,
)
from BalloonPoppingGymEnv.evaluation.results.utils import pack_for_submission


def _credential(text, field):
    # Some copied email clients escape underscores as ``\_``.
    normalized = text.replace("\\_", "_")
    match = re.search(rf"(?m)^\s*{re.escape(field)}\s*:\s*([^\s]+)\s*$", normalized)
    if match is None:
        raise ValueError(f"credentials file does not contain {field!r}")
    return match.group(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("credentials_file", type=Path, nargs="?")
    parser.add_argument("--dry-run", action="store_true",
                        help="Run official evaluation without credentials or packaging")
    parser.add_argument("--min-score", type=int, default=0,
                        help="Refuse to package a run below this measured score")
    parser.add_argument("--report", type=Path,
                        help="Write a credential-free evaluation receipt")
    args = parser.parse_args()

    eval_cfg = yaml.safe_load(args.config.read_text(encoding="utf-8-sig"))
    if args.dry_run:
        team_name = team_secret = None
    elif args.credentials_file is not None:
        credentials = args.credentials_file.read_text(encoding="utf-8-sig")
        if args.credentials_file.suffix.lower() == ".json":
            previous_submission = json.loads(credentials)
            team_name = previous_submission["team"]["name"]
            team_secret = previous_submission["team"]["secret"]
        else:
            team_name = _credential(credentials, "team_name")
            team_secret = _credential(credentials, "team_secret")
    else:
        try:
            team_name = os.environ["BPC_TEAM_NAME"]
            team_secret = os.environ["BPC_TEAM_SECRET"]
        except KeyError as error:
            raise ValueError(
                "provide a credentials file or set BPC_TEAM_NAME and BPC_TEAM_SECRET"
            ) from error

    if not args.dry_run:
        eval_cfg["team_name"] = team_name
        eval_cfg["team_secret"] = team_secret
    eval_cfg["leaderboard_submission"] = not args.dry_run

    configure_console_logging()
    agent_class = _load_agent_class(
        eval_cfg["agent_module_path"], eval_cfg["agent_class_name"]
    )
    env, _, scenario_parameters = evaluate_scenario(
        agent_class,
        agent_kwargs=eval_cfg["agent_kwargs"],
        agent_name=eval_cfg["agent_name"],
        scenario_number=eval_cfg["scenario_number"],
        render_mode=eval_cfg["render_mode"],
    )
    try:
        score = int(env._popped_count)
        if args.report is not None:
            safe_config = {key: value for key, value in eval_cfg.items()
                           if key not in {"team_name", "team_secret"}}
            receipt = {
                "evaluation": "unmodified official evaluate_scenario",
                "score": score,
                "seed": int(env.np_random_seed),
                "config": safe_config,
                "agent_sha256": hashlib.sha256(
                    Path(eval_cfg["agent_module_path"]).read_bytes()
                ).hexdigest(),
                "dry_run": args.dry_run,
                "uploaded": False,
            }
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
        if score < args.min_score:
            raise RuntimeError(f"Score {score} below required {args.min_score}; not packaged")
        if not args.dry_run:
            pack_for_submission(eval_cfg, env, scenario_parameters)
    finally:
        env.close()


if __name__ == "__main__":
    main()
