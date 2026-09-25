# Continuous timing and reachability experiments (2026-09-24)

## Evidence and research boundary

The verified reference is the standalone final-approach agent: Scenario 1,
seed 0 score 10; paired seeds 0/3 score 10/8. Its payload is preserved.

Primary research:

- Richter, Bry and Roy, polynomial trajectory optimization:
  https://groups.csail.mit.edu/rrg/papers/Richter_ISRR13.pdf
- Foehn, Romero and Scaramuzza, Time-Optimal Planning for Quadrotor Waypoint Flight:
  https://arxiv.org/abs/2108.04537

These motivate jointly choosing arrival times and trajectory derivatives.
Our implementation is a bounded minimum-jerk timing refinement for a rocket,
NOT their full quadrotor formulation and not a proof of time optimality.

Public workshop preparation points to the organizers' ws/coscup-2026 branch:
https://gist.github.com/hrchu/d78da1d978fab0c6ad4cdcdfa2abcd13
Its stated 7-point baseline concerns Scenario 0, not evidence of a high-scoring
Scenario 1 competition agent. Searches did not verify another team's public
winning source or a PPO-100M training recipe. The live leaderboard could not
be opened by the web tool; no current ranking is claimed.

Rechecked organizer clarification:
https://github.com/ARRC-Rocket/BalloonPoppingChallenge/discussions/165
Agents use only observations and given parameters; no stored route, target
order, inferred seed, hidden field or precomputed balloon trajectory is loaded.
Simulator, scoring, evaluator and submission packer are not modified.

## New agent implementations

`TimeAllocationAgent` refines each leg duration independently with SLSQP,
recomputing moving-target intercept positions and all joint knot derivatives
at every trial. A bounded trust region (0.6 to 1.5 times nominal durations,
also bounded by 0.3 seconds and the leg horizon) limits numerical jumps.
The objective is total time plus a small relative-duration regularizer.
Thrust, tilt, axis-rate, throttle slew, altitude and burn-time bounds remain
constraints. An initial 17-sample solve is followed by 65-sample refinement;
only candidates passing the existing dense physical validator are accepted.
These remain sampled reduced-model checks, not continuous guarantees.

The retimed route also proposes up to four currently released targets near
the continuous path. Insertion changes the number of waypoints and re-solves
arrival times. Failed solves preserve the old executable plan. The experiment
compares accepting only extra planned hits against accepting faster equal-count
routes too, with physical safety margins 0 and 0.015.

`ReachabilityChainAgent` changes only candidate screening: a constant-
acceleration estimate probes multiple possible arrival times, ranking required
force and tilt against availability. It either replaces the single 1.5-second
distance proxy, or reserves half the candidate slots for the previous nearest
heuristic. Joint chain planning and dense constraints still determine acceptance.
This estimate is not an exact reachable set and does not itself authorize a hit.

## Protocol

Fresh official-environment evaluations on paired development seeds 0 and 3;
these are not unseen seeds or an unbiased benchmark. CPU only, at most eight
concurrent workers on the authorized instructor server, BLAS/OMP one thread.
Each report records the main agent and imported agent source hashes before and
after execution, normal termination, wall time, score, route events and counters.
No leaderboard upload is performed.

Launch-time controls: 20, 22, 26, 28 seconds, otherwise unchanged reference.
Measured scores (seed 0 / seed 3): 8/8, 7/7, 8/8, 7/9 respectively.
None improves on the reference's 10-point maximum or paired mean of 9.

Reproduction examples:

```powershell
$env:OPENBLAS_NUM_THREADS = '1'
$env:MPLCONFIGDIR = Join-Path $PWD '.mplconfig'
.\.venv\Scripts\python.exe -m unittest tests.test_time_allocation tests.test_reachability_chain
.\.venv\Scripts\python.exe scripts/evaluate_guidance.py BalloonPoppingGymEnv/evaluation/configs/time_allocation.yaml --seeds 0 3 --set timing_safety=0 --set timing_only_more_hits=true --report BalloonPoppingGymEnv/evaluation/results/alloc_reproduction.json
.\.venv\Scripts\python.exe scripts/evaluate_guidance.py BalloonPoppingGymEnv/evaluation/configs/reachability_chain.yaml --seeds 0 3 --set nearest_fraction=0.5 --set branch_targets=12 --report BalloonPoppingGymEnv/evaluation/results/reach_reproduction.json
```

New numerical tests cover moving-target positions, C2 continuity, feasibility,
input nonmutation, invalid horizons, disabled behavior, velocity-aware ranking,
and released/failed/previously-routed candidate exclusion.

## Completed paired comparisons

| Variant | Seed 0 | Seed 3 |
| --- | ---: | ---: |
| Reference final approach | 10 | 8 |
| Timing, safety 0, extra hits only | 10 | 9 |
| Timing, safety 0, allow faster equal-count | 9 | 6 |
| Timing, safety 0.015, extra hits only | 10 | 7 |
| Timing, safety 0.015, allow faster equal-count | 9 | 7 |
| Reachability only, 8 candidates | 9 | 6 |
| Reachability only, 12 candidates | 9 | 6 |
| Half nearest / half reachability, 8 candidates | 10 | 6 |
| Half nearest / half reachability, 12 candidates | 10 | 6 |

The first 24 completed reports (including the eight launch controls) are in
`BalloonPoppingGymEnv/evaluation/results/timing_phase1_reports_20260924/`.
All 24 terminated normally, without truncation or source changes.

The selected timing variant scored **10, 8, 9, 9, 7, 8, 7, 6** on seeds 0–7,
against reference **10, 8, 9, 8, 7, 8, 7, 6**. Mean increased from 7.875 to 8.0.
The initial seed-0/3 diagnostics recorded respectively one/two extra planned
waypoints; actual scores increased by zero/one. Planned hits are not scores.
Increasing the insertion budget from four to eight candidates retained 10/9.
The cheaper four-candidate setting is therefore retained for reproduction.

No result here yet establishes 14 or 20 points. Reachability ranking and the
other timing settings are retained as experiments, not promoted defaults.

The eight follow-up reports are saved under
`BalloonPoppingGymEnv/evaluation/results/timing_validation_reports_20260924/`.
All passed the same termination/nontruncation/source-immutability checks.

## Standalone reproduction

Generated without overwriting any older agent:

```powershell
.\.venv\Scripts\python.exe scripts/build_chain_submission_agent.py --time-allocation --output BalloonPoppingGymEnv/agents/submission_time_allocation_v1.py
```

The builder refuses existing output paths. Standalone source SHA256:
`92085e793fe2b44714accd5fbf825bf0477f69d4ad0d097b102254cb45d3d885`.
Config: `BalloonPoppingGymEnv/evaluation/configs/submission_time_allocation_v1.yaml`.
The unmodified official evaluator independently reproduced **10 points, seed 0**.
Payload (not uploaded):

`BalloonPoppingGymEnv/evaluation/results/20260924T083109.930Z_TASTI_Cool_Ba_Malaew_09938a81f52f49e3a9e086132a48bb90_submission.json`

Receipt: `evaluation/results/submission_time_allocation_v1_receipt.json`.
The packer's optional online evaluator-integrity lookup was blocked by socket
permissions. Local git comparison confirmed unchanged official simulator,
evaluator, packer and verifier against bd9bb51. The old 10-point standalone
agent and its payload remain untouched.

58 selected unittest tests passed, including nine timing/reachability/bundle
tests. No claim is made about the full pytest suite (pytest is not installed).

The official verifier passed all **19** checks. The embedded source matches
the standalone file; no static source-review flags were raised. Audit:
`evaluation/results/submission_time_allocation_v1_audit.json`.
Payload SHA256:
`0f1d78c805630dd29c243d21003a765859e6c01e5775bca01d82cbdbb2d5d46b`.
This is evidence of format/trajectory/source consistency, not organizer
certification of rules compliance. In the paired seed-0 reports the old and
new agents both score 10 and record their last pop at 52.83 seconds, so no
leaderboard score or tie-break advantage is claimed for this new payload.

## Additional exploration and handoff

Without further tuning, the selected timing agent scored **9, 9, 6, 8, 6, 8,
8, 6** on additional seeds 8–15. Across seeds 0–15 its mean is 7.75, maximum
10; the old reference was not evaluated on the added eight seeds in this
experiment, so there is no paired improvement claim for that set.
Reports: `evaluation/results/timing_extra_reports_20260924/`.
All eight runs terminated normally, without truncation or source changes.

This work completed 40 remote episodes plus one local official reproduction.
All remote jobs finished, the SSH session was closed, and no GPU was used.
The current evidence supports a small eight-seed paired improvement, not
14/20 points or a winning score. Nothing was uploaded to the leaderboard.
