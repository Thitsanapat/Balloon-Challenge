"""Build the credential-free source archive for qualification review."""

import ast
import hashlib
import json
from pathlib import Path
import re
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

import yaml

from scripts.build_scenario4_joint_submission_agent import build_source
from scripts.package_finalnight_joint import EXTRA
from scripts.package_joint_candidate_review import FILES
from scripts.qualification_run import DEFAULT_CONFIG


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "review/TASTI_Cool_Ba_Malaew_qualification_source_20261010.zip"
AGENT = ROOT / "BalloonPoppingGymEnv/agents/submission_scenario4_pure_joint_v2.py"
EXPECTED_AGENT_SHA256 = (
    "9ff7f93d21a945d0479289be57bd893d8f33ab169fb1a6bfeb153ea442b6efb9"
)
LIVE_MANIFEST = ROOT / "review/qualification_live_run_manifest.json"
RECEIPTS = (
    "qualification/receipts/20261010T111933Z_seed1497481877_462484df_official_receipt.json",
    "qualification/receipts/20261010T113157Z_seed2796897388_db8a5afe_official_receipt.json",
)
ADDITIONAL = (
    "LICENSE",
    ".gitmodules",
    "review/qualification_source_review_README.md",
    "review/qualification_live_run_manifest.json",
    "scripts/package_qualification_source_review.py",
) + RECEIPTS


def sha256(raw):
    return hashlib.sha256(raw).hexdigest()


def checked_files():
    if sha256(AGENT.read_bytes()) != EXPECTED_AGENT_SHA256:
        raise RuntimeError("Live agent hash differs from both qualification receipts")
    if AGENT.read_text(encoding="utf-8") != build_source():
        raise RuntimeError("Live agent no longer matches the deterministic builder")
    expected_config = ROOT / (
        "BalloonPoppingGymEnv/evaluation/configs/"
        "submission_scenario4_pure_joint_launch42.yaml"
    )
    if DEFAULT_CONFIG.resolve() != expected_config.resolve():
        raise RuntimeError("Qualification launcher no longer selects the reviewed config")
    config = yaml.safe_load(expected_config.read_text(encoding="utf-8-sig"))
    live = json.loads(LIVE_MANIFEST.read_text(encoding="utf-8"))
    if (live["agent_sha256"] != EXPECTED_AGENT_SHA256 or
            live["agent_class"] != config["agent_class_name"]):
        raise RuntimeError("Live-run manifest does not match the selected policy")
    expected_runs = {(item["seed"], item["score"]) for item in live["evaluations"]}
    actual_runs = set()
    for name in RECEIPTS:
        receipt = json.loads((ROOT / name).read_text(encoding="utf-8"))
        if (receipt["agent_sha256"] != EXPECTED_AGENT_SHA256 or
                receipt["config"]["agent_kwargs"] != config["agent_kwargs"] or
                receipt["config"]["agent_class_name"] != config["agent_class_name"] or
                receipt["config"]["scenario_number"] != 4 or
                receipt["dry_run"] is not False):
            raise RuntimeError(f"Live receipt does not match reviewed policy: {name}")
        actual_runs.add((receipt["seed"], receipt["score"]))
    if actual_runs != expected_runs:
        raise RuntimeError("Live receipt seed/score pairs differ from the manifest")

    names = set(FILES + EXTRA + ADDITIONAL)
    pending = list(names)
    while pending:
        name = pending.pop()
        if not name.endswith(".py"):
            continue
        tree = ast.parse((ROOT / name).read_text(encoding="utf-8-sig"))
        for node in ast.walk(tree):
            if (isinstance(node, ast.ImportFrom) and node.module and
                    node.module.startswith("BalloonPoppingGymEnv.agents.")):
                dependency = node.module.replace(".", "/") + ".py"
                if dependency not in names:
                    names.add(dependency)
                    pending.append(dependency)

    contents = {}
    credential_pattern = re.compile(
        rb"(?i)(?:team_secret|BPC_TEAM_SECRET)\s*[:=]\s*[\x22\x27]?"
        rb"[a-f0-9]{32,}"
    )
    for name in sorted(names):
        path = ROOT / name
        if (path.is_symlink() or not path.is_file() or
                not path.resolve().is_relative_to(ROOT)):
            raise ValueError(f"Invalid archive input: {name}")
        raw = path.read_bytes()
        if credential_pattern.search(raw):
            raise ValueError(f"Credential-like value detected in {name}")
        if name.endswith("_submission.json"):
            raise ValueError(f"Submission JSON must not be archived: {name}")
        contents[name] = raw
    return contents, actual_runs


def archive_entry(name, raw):
    info = ZipInfo(name, date_time=(2026, 10, 10, 0, 0, 0))
    info.compress_type = ZIP_DEFLATED
    info.external_attr = 0o100644 << 16
    return info, raw


def main():
    contents, actual_runs = checked_files()
    manifest = {
        "archive_purpose": "Qualification source-code review",
        "challenge_release": "v0.2.2",
        "agent_sha256": EXPECTED_AGENT_SHA256,
        "qualification_seed_scores": [
            {"seed": seed, "score": score} for seed, score in sorted(actual_runs)
        ],
        "files": {name: sha256(raw) for name, raw in contents.items()},
        "excluded": ["credentials", "team secret", "submission JSON", "trajectories"],
    }
    manifest_raw = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    manifest_name = "review/qualification_source_manifest.json"
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    with ZipFile(OUTPUT, "x") as archive:
        for name, raw in contents.items():
            archive.writestr(*archive_entry(name, raw))
        archive.writestr(*archive_entry(manifest_name, manifest_raw))
    with ZipFile(OUTPUT) as archive:
        if archive.testzip() is not None:
            raise RuntimeError("Archive CRC check failed")
        for name, digest in manifest["files"].items():
            if sha256(archive.read(name)) != digest:
                raise RuntimeError(f"Archive hash mismatch: {name}")
        if archive.read(manifest_name) != manifest_raw:
            raise RuntimeError("Embedded manifest mismatch")
        forbidden = [name for name in archive.namelist()
                     if name.endswith("_submission.json")]
        if forbidden:
            raise RuntimeError(f"Submission JSON included unexpectedly: {forbidden}")
    print(OUTPUT)
    print("SHA256=" + sha256(OUTPUT.read_bytes()))
    print(f"{len(contents)} source/review files; receipts={len(RECEIPTS)}; secrets=0")


if __name__ == "__main__":
    main()
