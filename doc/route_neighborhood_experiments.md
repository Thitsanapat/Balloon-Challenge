# Online whole-route neighborhood experiments (2026-09-23)

## Hypothesis and implementation

The frozen submission scores 10 on scenario 1, seed 0. Previous capture-region,
short-leg, clustering and near-path insertion experiments did not beat it.
`RouteNeighborhoodAgent` now searches **target sequences**, not only the shape
of a fixed route. It starts with the observation-only joint-derivative beam plan.

Proposals include inserting a currently released balloon between waypoints,
appending a tail hit, substituting a nearby target, swapping targets, relocating
a target, and scaling arrival durations by 0.95, 1.0 or 1.05. All arrival
positions are predicted anew from the current observed position and velocity.
Every trial re-solves all internal velocities and accelerations together.
Reduced-model thrust, tilt, attitude-rate, throttle-slew and ground constraints
are checked at 17 samples per leg, then at 65 before adopting a plan. These
sampled checks are not continuous-time proofs and do not replace simulation.

A bounded beam preserves distinct target sequences. The objective is more
planned hits first, shorter completion time second. `only_more_hits=true`
requires an increased planned count before changing the original route;
`false` also accepts a shorter plan with equal count. Plans can differ from
actual scores because prediction and tracking are approximate. No change is
adopted if the dense feasibility check fails. Search runs on newly built routes,
not on every sensor update; committed flight tracking remains unchanged.

## Reproduction

From the repository root, using the same v0.2.1 simulator and pinned ActiveRocketPy:

```powershell
$env:OPENBLAS_NUM_THREADS = '1'
$env:MPLCONFIGDIR = Join-Path $PWD '.mplconfig'
.\.venv\Scripts\python.exe -m unittest tests.test_route_neighborhood tests.test_route_optimization tests.test_chain_bundle tests.test_chain_launch tests.test_opportunistic_chain tests.test_capture_chain tests.test_submission_source_audit -v
.\.venv\Scripts\python.exe scripts/evaluate_guidance.py BalloonPoppingGymEnv/evaluation/configs/route_neighborhood.yaml --seeds 0 1 2 3 --set neighborhood_budget=600 --set only_more_hits=true --report BalloonPoppingGymEnv/evaluation/results/neighborhood_hits.json
.\.venv\Scripts\python.exe scripts/evaluate_guidance.py BalloonPoppingGymEnv/evaluation/configs/route_neighborhood.yaml --seeds 0 1 2 3 --set neighborhood_budget=1600 --set only_more_hits=false --report BalloonPoppingGymEnv/evaluation/results/neighborhood_time.json
```

Budget limits count reduced-model candidate solves per search; final dense
checks and the original beam search are additional work. Defaults: 3 search
rounds, beam of 4 distinct orders, 2 nearby candidates per insertion/substitution.
An independent local disabled-search control uses `neighborhood_budget=0`.
The selected seeds 0–3 have been used in development before, so this is a paired
development comparison, **not** an unbiased generalization estimate.

## Rules boundary

The agent reads only `observation` and `given_parameters`; it imports agent-side
mathematics and does not read the environment, files, replay data or seeds.
Sequences are constructed online from released targets. The cached polynomial
matrices depend on durations and a cost weight, not on balloon layouts.
Seed selection occurs only in the external development evaluation harness.
Official simulator, evaluator, verifier and submission packer are unchanged.

Reviewed organizer clarification:
https://github.com/ARRC-Rocket/BalloonPoppingChallenge/discussions/165

This is an implementation boundary review, not organizer approval. The
organizers retain final interpretation of the rules. The new agent is an
experiment, not yet a promoted or uploaded competition submission.

## Results: neighborhood search and receding review

| Agent / setting | Seed 0 | Seed 1 | Seed 2 | Seed 3 | Mean |
|---|---:|---:|---:|---:|---:|
| Previously measured reference | 10 | 8 | 9 | 7 | 8.50 |
| Neighborhood, budget 600, only more planned hits | 10 | 8 | 9 | 7 | 8.50 |
| Neighborhood, budget 1600, also accept shorter equal-count plans | 10 | 8 | 6 | 6 | 7.50 |
| Fresh route review every 3 seconds, only more planned hits | 9 | 8 | 8 | 8 | 8.25 |
| Fresh route review every 6 seconds, only more planned hits | 10 | 8 | 9 | 7 | 8.50 |

All 16 remote episodes terminated normally, without truncation. Imported agent
source fingerprints were unchanged throughout each run. The local disabled
neighborhood control reproduced 10, including the reference pop times.

Across the eight neighborhood runs, retained diagnostics recorded 10,673 trial
solves and 99 coarse-feasible candidates, but no additional planned hits were
adopted. These counters exclude the discarded independent launch-direction
comparisons, so they are not a complete count of all CPU work. The permissive
variant adopted shorter equal-count plans, but lost actual hits in seeds 2/3.
Do not equate a shorter planned duration with increased actual score.

`RecedingChainAgent` is a separate follow-up: review the entire route against
newly released observations, while avoiding changes within 1.5 seconds of the
next planned hit. Each review solves a new beam plan and accepts only more
planned hits by default. Rejecting a proposal restores all prior controller
state, including acceleration and the current reference, while retaining work
counters. At 3-second review intervals, all four seeds adopted one increased
planned-count route, but the paired mean decreased. Six-second review intervals
adopted no changes. Neither setting replaces the reference.

Reports: `evaluation/results/neighborhood_reports_20260923/` and
`evaluation/results/receding_reports_20260923/`, both under `BalloonPoppingGymEnv`.
Local control: `evaluation/results/neighborhood_disabled_s0.json`.

Reproduce the follow-up by evaluating `evaluation/configs/receding_chain.yaml`
with `--set route_review_interval=3` or `=6`, seeds 0–3 (same script as above).

## Follow-up: specific-thrust effort in the spline objective

The minimum-jerk curve is only one choice of arrival derivatives. Rejecting
that curve does not prove that the target order itself is physically impossible.
`EffortChainAgent` tests a different joint quadratic objective:

`integral (||jerk||^2 + effort_weight * ||acceleration - gravity - disturbance||^2) dt`

For fixed arrival times and positions, the remaining knot velocities and
accelerations are solved jointly by a linear system. The acceleration integral
is analytic in the normalized quintic coefficients. The affine gravity and
disturbance term is included in the right-hand side. The same sampled physical
constraints still apply afterwards. This objective is an effort **proxy**, not
a fuel-consumption model; it does not extend engine burn duration. Weight has
units of inverse seconds squared. Zero weight uses the original solver exactly.

The development beam planner now has a `_chain_curves` hook; its default still
calls the unchanged minimum-jerk solver. Frozen submission files are untouched.
Tests check zero-weight identity, waypoint/derivative continuity, reduced
specific-thrust effort, and the quadratic optimum against perturbations of
each solved arrival derivative. Remote weights 0.1 and 0.5 use seeds 0–3;
a second local zero-weight control checks end-to-end equivalence.

Reproduce by evaluating `evaluation/configs/effort_chain.yaml` with
`--set effort_weight=0.1` or `=0.5` (same evaluation script and seeds).

### Completed effort-model results (retrieved 2026-09-24)

| Effort weight | Seed 0 | Seed 1 | Seed 2 | Seed 3 | Mean |
|---|---:|---:|---:|---:|---:|
| 0.1 | 3 | 7 | 3 | 9 | 5.50 |
| 0.5 | 6 | 6 | 3 | 7 | 5.50 |

All eight episodes terminated normally, without truncation; imported source
fingerprints remained unchanged during each run. The second local control
(`effort_weight=0`) reproduced 10 points and the reference pop times exactly.
The additional cost improves its mathematical effort objective, but neither
tested weight improves the paired mean or exceeds the frozen best score.
Consequently the effort-cost variants are not promoted.

Reports: `evaluation/results/effort_reports_20260923/` (under
`BalloonPoppingGymEnv`), plus `evaluation/results/effort_disabled_s0.json`.
The transfer was interrupted after the simulations finished; it was resumed
on September 24. These are the original completed runs, not new evaluations.

Across this development batch there were 24 remote episodes and two local
disabled-feature controls, all completed. The related 26 unit tests passed.
Only CPU was used, with at most eight remote evaluation workers at once.
Best verified submission remains 10 points; no 14/20-point result is claimed.

The next hypothesis worth testing is constraint-aware derivative optimization:
solve for a physically feasible curve directly, rather than rejecting an
entire target order because its unconstrained minimum-cost curve is infeasible.
This is future work, not an implemented or measured improvement.

## Submission status

The frozen 10-point JSON was re-audited during this turn. Embedded source still
matches the standalone file, all 19 official verifier checks pass, and official
components match release `bd9bb51`. Receipt:
`evaluation/results/neighborhood_frozen_submission_audit.json`.
No new competition payload was generated or uploaded for the rejected variants.

## Scenario and scoring clarification (checked 2026-09-24)

The current official README specifies:

- Leaderboard uses Scenario 1; the top ten teams advance. Waiting for another
  scenario is not required to improve or submit this round's score.
- Score is the number of popped balloons. Ties use the time of the last pop.
  Fuel economy, route smoothness and the scenario number do not award bonus
  points; they matter only insofar as they improve actual hits or tie-break time.
- Scenario 2/3 were released for preparation according to the September 18
  organizer email supplied by the user. The local release contains both sets
  of parameters. Their scores are not added to the Scenario 1 leaderboard.
- Qualification is October 10: two live evaluations whose scores are summed,
  with newly drawn seeds. The README lists Scenario 4 parameters to be released
  September 26. Training against 2/3 can prepare for noise and actuator response,
  but does not substitute for those final qualification parameters.
- Published leaderboard deadline: September 26, 2026, 00:05 GMT+8, which is
  **September 25, 2026, 23:05 in Thailand (GMT+7)**.

Source (rules are explicitly subject to change):
https://github.com/ARRC-Rocket/BalloonPoppingChallenge/blob/main/README.md#competition-rules-subject-to-change

Priority: preserve a validated Scenario 1 submission before its deadline;
develop robustness on 2/3 without assuming that their scores transfer to the
leaderboard; verify the announced qualification parameters when released.
