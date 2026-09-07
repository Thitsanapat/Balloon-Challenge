"""Run an evaluation and create a leaderboard submission without saving secrets.

The credentials file is read at runtime.  Team credentials are kept out of the
tracked YAML configuration, while the official evaluator and submission packer
still receive the exact fields they require.
"""

import argparse
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
    args = parser.parse_args()

    eval_cfg = yaml.safe_load(args.config.read_text(encoding="utf-8-sig"))
    if args.credentials_file is not None:
        credentials = args.credentials_file.read_text(encoding="utf-8-sig")
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

    eval_cfg["team_name"] = team_name
    eval_cfg["team_secret"] = team_secret
    eval_cfg["leaderboard_submission"] = True

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
    pack_for_submission(eval_cfg, env, scenario_parameters)


if __name__ == "__main__":
    main()
