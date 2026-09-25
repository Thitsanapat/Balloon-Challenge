# Scenario 1: constrained extra-hit trajectory optimization (2026-09-24)

## Change

`ConstrainedChainAgent` retains the reference launch selection and beam route.
It proposes an additional currently observed, released balloon near that route,
then solves directly for the waypoint arrival velocities and accelerations.
The initial position/velocity/acceleration, waypoint positions, arrival times,
and C2 continuity are preserved by a polynomial derivative basis. This differs
from earlier timing-only SQP and unconstrained minimum-jerk/effort objectives.

SLSQP minimizes squared changes in scaled arrival derivatives, subject to
sampled thrust-to-mass, positive thrust, tilt, thrust-axis rate, throttle slew
and ground-clearance constraints. Constraints use only given physical
parameters and the agent's observation-based disturbance estimate. Failed
solves do not overwrite the current feasible plan. Final checks use 65 samples
per leg, with a second solve at that density if necessary. Sampled checks are
not a proof of continuous-time feasibility or of actual simulator hits.

No environment internals, seeds, cached balloon trajectories or prerecorded
controls are accessed by the agent. No official simulator/evaluator/packer
components are modified. The frozen 10-point submission remains separate.

## Bounded comparison

Scenario 1 only. Compare repair on newly built routes (`repair_interval=0`)
against periodic review (`repair_interval=3`), paired on development seeds 0–3.
Defaults: up to six nearby candidate balloons per review, proximity radius
18 m, 35 coarse solver iterations and up to 17 dense iterations per candidate.
Only one extra waypoint is accepted in a review, without extending the final
arrival time. Seeds 0–3 have been used before; this is not a held-out estimate.
Use CPU only, at most eight remote workers; no GPU allocation.

```powershell
$env:OPENBLAS_NUM_THREADS = '1'
$env:MPLCONFIGDIR = Join-Path $PWD '.mplconfig'
.\.venv\Scripts\python.exe -m unittest tests.test_constrained_chain -v
.\.venv\Scripts\python.exe scripts/evaluate_guidance.py BalloonPoppingGymEnv/evaluation/configs/constrained_chain.yaml --seeds 0 1 2 3 --set repair_interval=0 --report BalloonPoppingGymEnv/evaluation/results/constrained_i0.json
.\.venv\Scripts\python.exe scripts/evaluate_guidance.py BalloonPoppingGymEnv/evaluation/configs/constrained_chain.yaml --seeds 0 1 2 3 --set repair_interval=3 --report BalloonPoppingGymEnv/evaluation/results/constrained_i3.json
```

A local control uses `--set repair_candidates=0` to reproduce the reference.
Unit tests cover actual invalid-to-feasible repair, continuity, margin signs
against the existing physical validator, burn deadline rejection, failed-repair
fallback and successful adoption bookkeeping. They do not establish scores.

## Results

| Variant | Seed 0 | Seed 1 | Seed 2 | Seed 3 | Mean |
|---|---:|---:|---:|---:|---:|
| Previously measured reference | 10 | 8 | 9 | 7 | 8.50 |
| Constrained insertion on new routes | 10 | 8 | 9 | 7 | 8.50 |
| Constrained insertion reviewed every 3 seconds | 10 | 8 | 9 | 7 | 8.50 |

All eight episodes ended normally, without truncation. Source fingerprints
remained unchanged in every run. New-route-only wall times were 155–179 s;
periodic review took 156–205 s. Both variants accepted one additional planned
waypoint in seed 0, but the actual score did not increase. Other seeds accepted
none. The local disabled-search control reproduced 10 points and the original
pop times. Do not promote on planned hit count or solver convergence alone.

Reports: `BalloonPoppingGymEnv/evaluation/results/constrained_insertion_reports_20260924/`.
Control: `BalloonPoppingGymEnv/evaluation/results/constrained_disabled_s0.json`.

## Follow-up: apply constraints within target-sequence expansion

`ConstrainedBeamAgent` invokes the same derivative repair within beam search,
not after a route has already been selected. Only near-feasible candidates are
considered (worst normalized constraint margin at least -0.15). Each beam
search has at most 16 repairs, with 25 coarse iterations plus a possible dense
repair. Failed repairs return the original curve, which still must pass the
existing physical validator to enter the beam. No constraint is relaxed.

Compare starting repairs at depth 6 versus depth 8, keeping the reference
beam width, search depth, launch timing and controller unchanged. These are
planning hyperparameters, not hidden field or seed information. Both variants
use the same external seeds 0–3. This comparison tests whether early rejection
of a minimum-jerk curve is pruning useful target orders.

```powershell
.\.venv\Scripts\python.exe scripts/evaluate_guidance.py BalloonPoppingGymEnv/evaluation/configs/constrained_beam.yaml --seeds 0 1 2 3 --set beam_repair_depth=6 --report BalloonPoppingGymEnv/evaluation/results/constraint_beam_d6.json
.\.venv\Scripts\python.exe scripts/evaluate_guidance.py BalloonPoppingGymEnv/evaluation/configs/constrained_beam.yaml --seeds 0 1 2 3 --set beam_repair_depth=8 --report BalloonPoppingGymEnv/evaluation/results/constraint_beam_d8.json
```

| Repair starts at depth | Seed 0 | Seed 1 | Seed 2 | Seed 3 | Mean |
|---|---:|---:|---:|---:|---:|
| 6 | 10 | 8 | 9 | 7 | 8.50 |
| 8 | 9 | 8 | 9 | 8 | 8.50 |

Both variants performed successful constraint repairs, but neither improves
the four-seed paired mean or best score. Depth 8 gained one hit on seed 3 and
lost one on seed 0. All episodes completed without truncation, with unchanged
imported sources. Reports:
`BalloonPoppingGymEnv/evaluation/results/constrained_beam_reports_20260924/`.

## Follow-up: tracking margin for repaired routes

Feasible repaired curves can lie on a constraint boundary. To test whether
tracking headroom helps, `beam_repair_safety` tightens the normalized thrust,
tilt and axis-rate inequalities within the optimization. It ramps from zero
at the fixed initial state to the configured margin at the final waypoint.
The ground, minimum-force and throttle-slew constraints are not relaxed.
The optimizer result must pass both tightened margins and the original dense
physical validator. Safety zero preserves the prior repair behavior.

Compare safety margins 0.01 and 0.03 at repair depth 8, seeds 0–3. These numbers
are normalized constraint margins, not a blanket percentage of available fuel.
For example, 0.03 corresponds to 0.36 m/s^2 of additional terminal thrust-to-mass
headroom in the normalized thrust inequality. This is a hypothesis test of
tracking robustness, not a diagnosis established by the previous score drop.

Use `constrained_beam.yaml` with `--set beam_repair_depth=8` and
`--set beam_repair_safety=0.01` or `=0.03`. Source snapshots before and after
the safety extension are retained as `constrained_chain_source.tgz`,
`constrained_beam_source.tgz`, and `constrained_robust_source.tgz` in results.

| Safety margin, depth 8 | Seed 0 | Seed 1 | Seed 2 | Seed 3 | Mean |
|---|---:|---:|---:|---:|---:|
| 0.01 | 9 | 8 | 9 | 8 | 8.50 |
| 0.03 | 9 | 8 | 9 | 8 | 8.50 |

All eight safety-margin runs terminated normally with unchanged agent sources.
Runtime ranged from 157 to 316 seconds. Neither margin recovered the lost
seed-0 hit, so the tracking-headroom hypothesis did not produce a score gain.
Reports: `BalloonPoppingGymEnv/evaluation/results/constrained_robust_reports_20260924/`.

## Conclusion and retained artifacts

This batch completed **24 remote evaluations plus one local disabled-feature
control**, all on Scenario 1. The 34 related unit tests pass. No GPU was used;
all remote workers have finished and the SSH session is closed.

Constraint optimization demonstrably repaired curves that the unconstrained
planner rejected, and depth 8 gained one actual hit on seed 3. However, it lost
one hit on seed 0, leaving the paired mean at 8.50 and the best score at 10.
Therefore none of these variants replaces the frozen competition submission.
No new payload was generated or uploaded. The original JSON remains unchanged:

`20260923T074855.512Z_TASTI_Cool_Ba_Malaew_3d64859bc1cc4647876b30e2f8928efd_submission.json`

SHA256: `b6ff58aac9ad159357a1f33f155546bf2cb4d42a8636a609570b9c7c6a280afe`.
Official environment, evaluator, packer and verifier still compare unchanged
against release `bd9bb51`.

An observed failure worth diagnosing next: depth-8 seed 0 switches its late
route to balloon 34 followed by 12. Balloon 12 is missed; the evaluation's
closest observed distance is approximately 4.018 m versus the 1.5 m radius.
This is observation-sampled distance, not the official swept collision test.
The evidence does **not** yet separate balloon-prediction error from rocket
tracking error. Before further optimizer tuning, instrument that separation
using only observation history and the agent's own reference, then test the
identified error source. No 14/20-point result is claimed.
