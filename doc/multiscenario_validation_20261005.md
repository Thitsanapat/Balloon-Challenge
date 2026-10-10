# Frozen mixed-scenario PPO validation, 2026-10-05

## Training pilot completed

The authorized pilot stopped normally at its two-hour wall budget after
13,038 policy decisions (130,202 controlled physics steps; warmup excluded).
There were 35 completed training episodes, zero official truncations, and no
source mutation or error file. The process has exited and released its GPU.

| Scenario | Training episodes | Mean actual pops | Minimum | Maximum |
| --- | ---: | ---: | ---: | ---: |
| 0 | 1 | 10 | 10 | 10 |
| 1 | 5 | 8.4 | 8 | 9 |
| 2 | 4 | 8 | 7 | 9 |
| 3 | 9 | 8.222 | 6 | 10 |
| 4 | 16 | 3.25 | 0 | 7 |

These are stochastic training episodes from changing policies, not final-policy
validation, not proof of improvement, and not competition submission scores.

The final checkpoint and provenance logs are backed up locally in
`BalloonPoppingGymEnv/evaluation/results/multiscenario_final_snapshot_20261005.zip`.
Archive SHA-256 matches the remote copy:
`d3a9061d940ed087c2962ebd1d0aa907c81872f000c745b7cdaf05cd0cb6236d`.
Contained `final_model.zip` SHA-256:
`8aad07e2ac6b834601b1d992f0fbb3ce9bcce1d1bc845ea179623b9e98577b3b`.
No credentials are in the snapshot. This is a model backup, not submission JSON.

## Predeclared paired validation

- Scenario 4 only, seeds `1000000007`, `1111111111`, `1428571429`.
- All seeds are in the validation pool, disjoint from the training seed log.
  They were fixed before viewing results; do not replace low-scoring seeds.
- Baseline: `MultiScenarioResidualController`, default settings, zero residual.
  This is the same full-route joint-refresh controller used during training,
  not the older wind-profile-only controller.
- Candidate: `MultiScenarioPPOPolicyAgent`, frozen final checkpoint,
  deterministic prediction, identical controller with bounded learned residual.
- Fresh official environment per flight, unchanged physics/weather/RocketPy;
  no cached field or privileged state is supplied to the agent.
- CPU only (`CUDA_VISIBLE_DEVICES=''`), one flight at a time, native numerical
  thread limits set to one, process nice level 10. No GPU is needed for this
  small policy's inference. Six flights maximum, 1,200-second timeout per flight.
- Check checkpoint, manifest, training source and full runtime snapshot hashes
  before/after each flight. Stop on failure/truncation, keep failed seed visible.
- Reports separate actual pops from any training reward shaping. A win on three
  validation seeds is not sufficient for generalization or automatic promotion;
  an untouched holdout is still required.

Runner: `scripts/validate_multiscenario_pilot.py`. It saves a predeclared
`plan.json`, per-flight configs/logs/reports, `status.json`, and `summary.json`
only after all six flights succeed. Existing directories are not overwritten.

Remote root: `/home/rtcmspaceweather/balloon_multiscenario_20261004_v1/`.
Runtime remains in `source/`; only new evaluation helper files were added,
without overwriting the frozen training/runtime files. Validation results are
in `validation_20261005/`, launcher log `validation_20261005.log`.
Launch PID: `1528523`; verify the command before using a PID operationally.

Local runner/summary tests: 7 passed. Windows Application Control blocked
`torch/lib/shm.dll` for the local model tests; no security policy was changed.
The matching remote environment passed all 9 mixture, adapter and runner tests.
An additional local decision-cadence test passed (8 local tests including the
runner/summary suite): warmup does not call the policy, decisions remain ten
steps apart, and clearing a target's old residual does not shift the clock.
Official evaluator/environment diff against release `v0.2.2` is empty.
The local suite was subsequently expanded to 9 passing tests, including mocked
six-flight orchestration, CPU isolation, refusal to overwrite prior reports,
and fail-fast behavior. Scores printed by that mocked test are fixtures only,
not simulation results. No frozen remote runner/controller file was changed.

The final complete training log block at 12,928 decisions reports 500 optimizer
epochs, approximate KL 0.0002353, clip fraction 0, and explained variance 0.505.
These support training-health inspection only, not a score improvement claim.

## Result status

**Completed:** all three predeclared pairs terminated normally in 1,526.25 s.
Frozen source/runtime/model checks passed; no upload. Final scores:

| Seed | Zero-residual base | PPO | Difference |
| --- | ---: | ---: | ---: |
| 1000000007 | 5 | 3 | -2 |
| 1111111111 | 1 | 3 | +2 |
| 1428571429 | 2 | 5 | +3 |
| Mean per flight | 2.667 | 3.667 | +1 |

PPO won two pairs and lost one. This is a small validation result, not enough
to certify generalization or reach 20 pops. Full reports/configs/plan/summary
are now backed up in `results/multiscenario_validation_complete_20261005.zip`
under `BalloonPoppingGymEnv/evaluation/`. Remote/local SHA-256 match:
`bb8c1b4dc2204d2e55e2e73d1a1206acbce732a89c4fee71db9520bcd0fda134`.
The authorized next stage is documented in
[ten-seed holdout plan](multiscenario_holdout_20261005.md).

### Earlier first-pair snapshot

First pair completed on seed `1000000007`: **baseline 5, PPO 3** (delta -2).
Both terminated normally, with unchanged source/model checks. Baseline reset
took 143.98 s and flight/diagnostics 103.65 s (total 247.63 s). Candidate total
was 250.56 s. These are real deterministic validation flights, not training.

Baseline pop IDs: 15, 25, 84, 27, 36. Candidate: 15, 47, 71. The first hit was
at 50.92/50.93 s respectively, but the subsequent route diverged. Mean tracking
position error (`position_error_sum/tracking_steps`) was approximately
0.298/0.332 m; navigation RMS was 0.525/0.480 m. Thus the lower score is not
explained by a larger navigation RMS in this one flight; these diagnostics do
not establish a causal explanation for the policy's worse route outcome.

At that snapshot, the remaining two predeclared pairs were pending. Keep the zero-residual
controller; do not promote this PPO based on the pilot. No new training or upload
was started. Existing submission artifacts are untouched.

First-pair reports and predeclared plan are backed up locally in
`BalloonPoppingGymEnv/evaluation/results/multiscenario_validation_first_pair_20261005.zip`.
SHA-256: `a85de9f686fb73808c451ea9146a16bd1ea6fdf982f4d2059230f8903a518256`.
The saved status is a snapshot, not live completion of the remaining pairs.
