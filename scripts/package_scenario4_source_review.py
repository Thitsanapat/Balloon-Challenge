"""Package the reviewed Scenario 4 launch-42 agent without run data or secrets.

Only the explicit source allowlist below is archived. The official simulator,
evaluation results, and credential-bearing submission JSON are never included.
"""

import argparse
import hashlib
import os
from pathlib import Path
import re
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "review" / "scenario4_qualification_source_v2.zip"
SOURCE_FILES = (
    "BalloonPoppingGymEnv/agents/submission_scenario4_wind_profile_v1.py",
    "BalloonPoppingGymEnv/agents/submission_time_allocation_v1.py",
    "BalloonPoppingGymEnv/agents/scenario4_wind_profile_agent.py",
    "BalloonPoppingGymEnv/evaluation/configs/submission_scenario4_wind_profile_launch42.yaml",
    "scripts/build_scenario4_submission_agent.py",
    "scripts/run_submission.py",
    "scripts/package_scenario4_source_review.py",
    "doc/scenario4_qualification_work.md",
    "review/scenario4_qualification_source_README.md",
)
SECRET_ASSIGNMENT = re.compile(
    rb"(?im)^\s*(?:team_secret|BPC_TEAM_SECRET)\s*[:=]\s*['\"]?[a-z0-9]{16,}"
)


def checked_source_files():
    for name in SOURCE_FILES:
        if Path(name).suffix.lower() == ".json":
            raise ValueError(f"JSON is not allowed in source archive: {name}")
        source = ROOT / name
        if source.is_symlink() or not source.is_file() or ROOT not in source.resolve().parents:
            raise ValueError(f"Missing or unsafe source file: {name}")
        contents = source.read_bytes()
        if SECRET_ASSIGNMENT.search(contents):
            raise ValueError(f"Possible team credential in source file: {name}")
        yield name, contents


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replace", action="store_true",
                        help="Replace only the named v2 archive after inspecting the source")
    args = parser.parse_args()
    if OUTPUT.exists() and not args.replace:
        raise FileExistsError(f"Already exists: {OUTPUT}; pass --replace explicitly")

    sources = list(checked_source_files())
    config = dict(sources)[
        "BalloonPoppingGymEnv/evaluation/configs/submission_scenario4_wind_profile_launch42.yaml"
    ].decode("utf-8")
    if not re.search(r"(?m)^\s*launch_time:\s*42(?:\.0)?\s*$", config):
        raise ValueError("Expected the reviewed 42-second launch config")
    if "submission_scenario4_wind_profile_v1.py" not in config:
        raise ValueError("Config does not select the standalone source")

    staging = OUTPUT.with_name(OUTPUT.name + ".staging")
    if staging.exists():
        raise FileExistsError(f"Staging archive already exists: {staging}")
    try:
        with ZipFile(staging, "x", compression=ZIP_DEFLATED) as archive:
            for name, contents in sorted(sources):
                entry = ZipInfo(name, date_time=(2026, 10, 3, 0, 0, 0))
                entry.compress_type = ZIP_DEFLATED
                entry.external_attr = 0o100644 << 16
                archive.writestr(entry, contents)
        os.replace(staging, OUTPUT)
    finally:
        if staging.exists():
            staging.unlink()

    print(OUTPUT)
    print("sha256=" + hashlib.sha256(OUTPUT.read_bytes()).hexdigest())
    print(f"source_files={len(sources)}")


if __name__ == "__main__":
    main()
