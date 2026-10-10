"""Package reviewed guarded Scenario-4 source without results or credentials."""

import hashlib
import os
from pathlib import Path
import re
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "review" / "scenario4_guarded_innovation_source_v2.zip"
SOURCE_FILES = (
    "BalloonPoppingGymEnv/agents/submission_scenario4_guarded_innovation_v1.py",
    "BalloonPoppingGymEnv/agents/submission_scenario4_wind_profile_v1.py",
    "BalloonPoppingGymEnv/agents/submission_time_allocation_v1.py",
    "BalloonPoppingGymEnv/agents/scenario4_wind_profile_agent.py",
    "BalloonPoppingGymEnv/agents/scenario4_forecast_innovation_agent.py",
    "BalloonPoppingGymEnv/agents/scenario4_guarded_innovation_agent.py",
    "BalloonPoppingGymEnv/evaluation/configs/submission_scenario4_guarded_innovation_launch42.yaml",
    "scripts/build_scenario4_submission_agent.py",
    "scripts/build_scenario4_guarded_submission_agent.py",
    "scripts/run_submission.py",
    "scripts/package_scenario4_guarded_source_review.py",
    "tests/test_scenario4_guarded_innovation.py",
    "tests/test_scenario4_guarded_submission_build.py",
    "doc/scenario4_guarded_innovation_20261007.md",
    "review/scenario4_guarded_innovation_source_README.md",
)
SECRET_ASSIGNMENT = re.compile(
    rb"(?im)^\s*(?:team_secret|BPC_TEAM_SECRET)\s*[:=]\s*['\"]?[a-z0-9]{16,}"
)


def checked_source_files():
    for name in SOURCE_FILES:
        if Path(name).suffix.lower() == ".json":
            raise ValueError(f"JSON not allowed in source archive: {name}")
        source = ROOT / name
        if source.is_symlink() or not source.is_file() or ROOT not in source.resolve().parents:
            raise ValueError(f"Missing or unsafe source file: {name}")
        contents = source.read_bytes()
        if SECRET_ASSIGNMENT.search(contents):
            raise ValueError(f"Possible team credential: {name}")
        yield name, contents


def main():
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite existing archive: {OUTPUT}")
    files = list(checked_source_files())
    staging = OUTPUT.with_name(OUTPUT.name + ".staging")
    if staging.exists():
        raise FileExistsError(f"Refusing to overwrite staging archive: {staging}")
    try:
        with ZipFile(staging, "x", compression=ZIP_DEFLATED) as archive:
            for name, contents in sorted(files):
                entry = ZipInfo(name, date_time=(2026, 10, 7, 0, 0, 0))
                entry.compress_type = ZIP_DEFLATED
                entry.external_attr = 0o100644 << 16
                archive.writestr(entry, contents)
        os.replace(staging, OUTPUT)
    finally:
        if staging.exists():
            staging.unlink()
    print(OUTPUT)
    print("sha256=" + hashlib.sha256(OUTPUT.read_bytes()).hexdigest())
    print(f"source_files={len(files)}")


if __name__ == "__main__":
    main()
