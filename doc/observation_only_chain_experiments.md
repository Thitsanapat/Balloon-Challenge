# Observation-only chain guidance — 2026-09-23

## Scope and rules

All results on this page use fresh scenario-1 simulations from the official
v0.2.1 environment, not the cached-field development harness used in older
experiment notes. Do not compare old cached scores directly with these scores.
The environment, evaluator, submission packer and verifier have no diff against
official release commit `bd9bb51`.

The submitted agent only consumes `given_parameters` and observations. It does
not receive an evaluation seed, load balloon fields, use public replays, read
simulator internals, mutate observations, or change physical parameters. Seed
selection is confined to the external evaluation harness. No PPO training was
performed for these agents. Organizer clarification:
<https://github.com/ARRC-Rocket/BalloonPoppingChallenge/discussions/165>.
These checks are not a substitute for the organizers' final rules judgment.

## What changed

The chain planner searches multiple prospective balloon intercepts before
executing the first leg. Adding a target re-solves the arrival velocity and
acceleration at **every** waypoint jointly. The objective is minimum integrated
squared jerk, using piecewise quintic polynomials with continuous position,
velocity and acceleration. Endpoint velocities are not forced to zero.

Candidate chains are checked against predicted thrust/mass, throttle slew,
axis turn rate, tilt and ground clearance. The winning chain is checked with
65 samples per segment. These are approximate planning constraints, not a proof
of continuous feasibility; the unchanged 6-DOF simulator determines the result.
After a hit, the next intercept deadline is retained instead of throwing away
the continuation. Targets are predicted from current observed position and
velocity, and feedback continually corrects the plan.

`ChainLaunchAgent` generates candidate rail directions from currently released
balloons, compares full online chains, and chooses a direction before launch.
It runs each candidate on a copy of its own state, then restores the original
state and initializes the attitude estimate to the selected rail direction.
The separate `submission_chain_launch_v1.py` bundle includes this experiment;
the older `submission_chain_v1.py` bundle is preserved unchanged.

## Measured comparison

| Scenario-1 seed | Old two-intercept baseline | Conservative chain | Agile chain |
| --- | ---: | ---: | ---: |
| 0 | 7 | 8 | 9 |
| 1 | 7 | 7 | 8 |
| 2 | 5 | 7 | 9 |
| 3 | 4 | 8 | 9 |
| 4 | 5 | 8 | 8 |
| 5 | 6 | 7 | 7 |
| Mean | 5.67 | 7.50 | 8.33 |

These are development evaluations, not a held-out statistical claim. Seed 0
and 1 guided tuning; seeds 2–5 were subsequently used for checks. Qualification
scenarios and independently drawn seeds still need separate evaluation.

Conservative chain settings: launch 24 s, replan every 0.4 s, beam width 12,
14 search levels, 8 candidate targets, 0.5 s duration grid, zero terminal penalty.

Other completed experiments:

- Finer 0.25 s duration grid: 5–6 points on seeds 0/1; not promoted.
- Wider beam (24) and 12 branches, launches 20/24/28/32 s: 5–8 points.
- Higher tilt/turn allowance, 75 degrees and 1.2 rad/s, attitude frequency 4:
  **9 / 8 points** on seeds 0/1 on the remote machine. These are agent planning
  and tracking settings, not edits to simulator actuator limits.
- Online rail selection plus the agile chain: **10 / 8 points** on seeds 0/1
  on the remote machine. Additional seeds 2/3 scored **9 / 7**. On those four
  seeds the mean is 8.5, versus 8.75 for the agile chain without rail selection:
  the best single score improved, but generalization did not improve overall.
- Increasing the turn-rate planning limit further to 1.5 rad/s: 9/8/8 on seeds
  0/1/2; 2.0 rad/s: 8/7/8. More aggressive planning is not uniformly better.
- Launch-direction planner with launch times 20/28/32 s: respectively 7/8,
  7/7 and 9/8 on seeds 0/1. None beat the 24-second, 10-point configuration.
- Natural/free-terminal momentum planning: up to 6 points in tested cases.
- Vertical powered-ascent model: up to 6 points in tested cases.
- Coverage MPC: up to 5 points in tested cases.
- Timing SQP refinement of the chain: 7 points on seed 0, below the 8-point
  chain reference; not promoted.
- Acceleration-ramp beam: 4 points on seed 0; not promoted.
- Original two-intercept baseline across seeds 0–76: no result above 7 points.

No experiment here has verified 14 or 20 points.

## Reproduction and artifacts

Frozen standalone agent: `BalloonPoppingGymEnv/agents/submission_chain_v1.py`.
SHA-256: `9b79b8643e8a32cad2fda537a610ef8b2bc1aaedcd2e0ba2c71dd21fa8b5e697`.
Generator: `scripts/build_chain_submission_agent.py`.

The conservative standalone config reproduced **8 points** locally through the
unmodified official evaluator and packer. Its payload is:

`evaluation/results/20260923T022239.546Z_TASTI_Cool_Ba_Malaew_ed5e6184de4f495bbe229fe8f4b06025_submission.json`

All official verifier checks passed: canonical parameters, balloon trajectories
and release times, recorded velocity/path consistency, and all 8 claimed hits.
The worst closest approach among those hits was 0.670 m, within the 1.5 m radius.
The official online evaluator integrity check was rerun with network access
after the sandboxed packer's advisory check could not connect; it completed
without warning. This verifier does not itself replay the rocket controller;
the fresh local evaluator run is the independent reproduction evidence.

From the repository root (PowerShell):

```powershell
.\.venv\Scripts\python.exe scripts/run_submission.py BalloonPoppingGymEnv/evaluation/configs/submission_chain_v1.yaml --dry-run --min-score 8 --report BalloonPoppingGymEnv/evaluation/results/chain_reproduction.json
.\.venv\Scripts\python.exe scripts/run_submission.py BalloonPoppingGymEnv/evaluation/configs/submission_chain_agile_v1.yaml --dry-run --min-score 9 --report BalloonPoppingGymEnv/evaluation/results/agile_reproduction.json
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_route_optimization.py
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_powered_ascent.py
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_chain_launch.py
```

The agile chain reproduced **9 points** locally with the official evaluator and
packer. All official verifier checks passed, including 9 reachable hits (worst
closest approach 0.218 m). The report is
`evaluation/results/submission_chain_agile_v1_verify.log`. Payload:

`evaluation/results/20260923T074356.012Z_TASTI_Cool_Ba_Malaew_6a7f06d28fda4a6993004dd82a11c056_submission.json`

The launch-selection standalone bundle reproduced **10 points** locally with
the official evaluator and packer. Payload (not uploaded):

`evaluation/results/20260923T074855.512Z_TASTI_Cool_Ba_Malaew_3d64859bc1cc4647876b30e2f8928efd_submission.json`

Receipt: `evaluation/results/submission_chain_launch_v1_receipt.json`.
Configuration: `evaluation/configs/submission_chain_launch_v1.yaml`.
All official verifier checks passed. The 10 claimed hits are reachable, with
worst closest approach 0.392 m against the 1.5 m radius; regenerated balloon
positions match exactly. See `evaluation/results/submission_chain_launch_v1_verify.log`.
The online evaluator-integrity check also completed without warning after being
rerun with network access.
Build its separate standalone source with
`python scripts/build_chain_submission_agent.py --with-launch`.
SHA-256: `86795ef3e0afdf17dc894b99a57dc6082446ef4bf674a745ce299eabea26aa57`.

Twelve added numerical/bundle tests pass across `test_route_optimization.py`,
`test_powered_ascent.py`, `test_chain_launch.py` and `test_chain_bundle.py`.
The bundle tests compare chain coefficients and rail candidates against the
development modules, and ensure the submission imports no custom agent module.

All 154 remote report files were copied to
`evaluation/results/remote_guidance_20260923/`; the transfer archive is
`evaluation/results/guidance_reports_20260923.tgz`. These files retain settings,
seeds, scores, wall times and (for later runs) route/launch diagnostics. Reports
from exploratory development are not all provenance-complete: older wrapper
versions recorded the entrypoint hash only, sometimes after the run, rather than
hashing every imported dependency. For reproducibility rely on the frozen
standalone source, official local receipts and embedded submission source.
All remote experiment jobs have finished; no training job is left running.

Experiments use CPU workers only (up to eight on the authorized instructor
machine); zero Tesla GPUs were used. Submission payloads contain team credentials:
keep them private and do not commit them. Nothing was uploaded to the leaderboard.
