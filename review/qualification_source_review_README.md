# TASTI_Cool_Ba_Malaew — Scenario 4 qualification source review

This archive contains the source used for both Balloon Popping Challenge
qualification evaluations on 10 October 2026. It is a source-review archive,
not a leaderboard submission JSON.

## Exact evaluated policy

- Official challenge release: `v0.2.2`, Scenario 4.
- Entry class: `Scenario4PureJointRefreshAgent`.
- Live source: `BalloonPoppingGymEnv/agents/submission_scenario4_pure_joint_v2.py`.
- Live config: `BalloonPoppingGymEnv/evaluation/configs/submission_scenario4_pure_joint_launch42.yaml`.
- Agent SHA-256:
  `9ff7f93d21a945d0479289be57bd893d8f33ab169fb1a6bfeb153ea442b6efb9`.
- No learned weights or GPU are required at runtime.

The two credential-free official receipts report 7 pops on seed `1497481877`
and 3 pops on seed `2796897388`. Both receipts contain the same agent hash and
the complete public agent configuration. They are included only to identify
the reviewed live runs. Submission JSON files are deliberately excluded
because they contain the team secret.

## Reproduction

1. Check out the organizer repository at tag `v0.2.2` with its recorded
   ActiveRocketPy submodule.
2. Overlay this archive at the repository root.
3. Create the environment from `uv.lock`, or follow the official v0.2.2 setup
   using `requirements.txt`. The recorded lockfile is preferred.
4. Set `scenario.random_seed` in the official Scenario 4 parameter file to the
   seed being reproduced.
5. Run the official evaluator through:

   ```text
   python -m scripts.run_submission BalloonPoppingGymEnv/evaluation/configs/submission_scenario4_pure_joint_launch42.yaml --dry-run
   ```

   `--dry-run` performs the flight without credentials or submission
   packaging. Remove it only under the organizer's normal credentialed
   submission procedure.

The single-file live agent imports only NumPy, SciPy, Python standard-library
modules, and the organizer's `BaseAgent`. The modular sources and deterministic
builder are also included to show how the standalone source was assembled.
Running the builder to a new unused path must reproduce the live agent bytes:

```text
python -m scripts.build_scenario4_joint_submission_agent --output rebuilt_agent.py
```

The runtime agent uses observations and public `given_parameters`; it does not
read the random seed, saved trajectories, future balloon motion, simulator
internals, or evaluator diagnostics. Offline telemetry and rejected
experimental policies are not required for either live evaluation.

## Integrity and privacy

`review/qualification_source_manifest.json` inside the ZIP lists the SHA-256
of every bundled file. The packer verifies the live receipts, deterministic
builder output, default config, archive CRC, and all listed hashes. The archive
contains no team secret, credentials file, or `_submission.json`.
