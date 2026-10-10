# Route-selection data collection: shadow parity pilot

## Purpose

The next experiment is a learned route selector, not another expansion of the
residual-control PPO. Before changing flight choices, collect the actual
planner decisions and verify that recording them has no behavioral effect.
The ten-seed-tested parent remains the reference; this collector is not a new
scoring agent or a submission.

## Implemented

- `BalloonPoppingGymEnv/training/route_shadow.py`: an external observer called
  only after the real agent's `get_action` returns and after each environment
  step. This avoids collecting the hypothetical launch-direction searches that
  deepcopy and restore controller state. The observer never enters that state,
  invokes a planner, reads environment internals, or draws random numbers.
- Records the existing 12/9-dimensional all-visible-balloon features, active
  target, committed deadline, filtered navigation, and route metadata. Hidden
  and nonfinite balloon rows are masked/zeroed. Seed is external provenance,
  not a policy feature. Wind features here are raw observations, not a claim
  to reproduce the controller's corrected prediction.
- Every selected route is described by prefixes of lengths 1 through N, so
  short and long routes are available for later proposal representation work.
  These are **not newly solved alternative trajectories** and are not a beam
  frontier dump. Their `actual_hits` remains null; their inherited feasibility
  has not been revalidated. Stale chain metadata is excluded.
- Executed first-leg outcomes: observed hit, latest-deadline miss, cancellation
  on target switch, or censoring when the episode ends before its deadline.
  A later hit after an earlier missed deadline can belong to another attempt;
  missed deadlines are not proof that a balloon can never be hit.
- A refresh of the same target stays in one attempt while updating its actual
  committed deadline. Candidate masks are retained honestly; a blocked active
  target is not an eligible categorical imitation label.
- `scripts/collect_route_shadow.py`: frozen-source/checkpoint checks and two
  fresh official flights per seed in separate processes. Gate compares hashes
  of **all actions and all observations**, step count, score, final time and
  normal termination, not merely equal scores. Failures stop collection and
  are recorded; existing result directories cannot be overwritten.
- `scripts/audit_route_shadow.py`: validates parity and dataset hash, schema,
  masks, outcomes, counterfactual-null labels and episode completeness. Groups
  refreshes by executed attempt and selects the first decision for imitation
  so repeated updates do not create spurious independent samples. Cancelled
  and censored attempts are excluded from binary deadline-outcome labels.

No existing flight controller, simulator, or official evaluator was edited.
The attention model and across-depth proposal selector remain untrained and
disabled in flight. Capturing/re-solving other beam-frontier candidates and
learning their route values are still separate subsequent tasks.

## Pilot provenance

- Scenario 4, predeclared training-pool seed `261006001`; not a held-out score
  test and not selected by inspecting fields or scores.
- Frozen parent: `pilot/final_model.zip`, SHA-256
  `8aad07e2ac6b834601b1d992f0fbb3ce9bcce1d1bc845ea179623b9e98577b3b`.
- Host project: `/home/rtcmspaceweather/balloon_multiscenario_20261004_v1/`.
- New isolated runtime: `source_route_shadow_20261006/`; earlier snapshots
  remain untouched. Results: `route_shadow_20261006/`; outer log:
  `route_shadow_20261006.log`.
- CPU only, one worker, one numerical thread, nice level 10, GPUs hidden.
  Per-flight timeout 1,500 seconds. No model training or upload.
- Started runner PID `1822471`; verify its command before acting on a PID.
- Pilot runner SHA-256:
  `7ca28e6817d107f01bb06c2961c43260a3524c2c5133486cb033275dc79923a1`.
- Collector SHA-256:
  `855b0a5ee947fc2de85545c1ca85e6df7ee9665554f4654d29ee3af578376f9d`.

Local command tests: 18 passed (7 collector/gate, 4 data-audit, 3 set-feature,
3 proposal-selection, 1 frozen policy-cadence). Remote CPU: 13 passed before
launch (7 collector/gate, 3 set-feature, 3 attention-tensor), plus 3 data-audit
tests subsequently. The extra fourth audit test was added locally after the
remote test copy. Compilation passed; official evaluator and
environment have no diff. These unit tests are not score evidence.

## Status and acceptance

**Completed: exact parity passed; dataset audit passed.** Both fresh flights
scored 6, terminated normally at 80.37 simulated seconds after 8,037 steps,
with no truncations. Baseline wall time: 265.202 s; shadow: 265.305 s. This
single pair is not a reliable overhead benchmark.

Both action streams SHA-256:
`fefd4809853d525b5d0ea3defd2bccfd10d2d9d8243125c594986ef3563e84e8`.
Both observation streams SHA-256:
`5e25a6a0d6348d3fb26973e30e569ca1470aa82e36006db58b8ad0a169714e3b`.

Collected 33 decisions/refreshes, grouped into 7 executed first-leg attempts:
5 observed target hits and 2 deadline misses, no cancellations/censored legs.
The **6 official pops and 5 labelled active-target hits are different metrics**:
the collector attributes only pops of its currently active first-leg attempt,
not every pop throughout the flight. Never silently use these labels as full
route reward. All 233 recorded prefix descriptions remain unlabelled; prefixes
are correlated descriptions of the selected route, not 233 tested flights.
Only 7 first decisions (one per attempt, all from the same field and therefore
still correlated) are eligible for imitation here.
One field is a plumbing test, insufficient diversity for model training or
evidence of scoring gains. No new model was trained or promoted.

Dataset SHA-256:
`b01f5d8d1c1136578b35e2890a85b66fda39992cfb052d2fd8479923437e55bd`.

Data, provenance, exact parity results, audit and collector source are backed
up locally in
`BalloonPoppingGymEnv/evaluation/results/route_shadow_complete_20261006.zip`.
Remote/local archive SHA-256 agrees; ZIP integrity test passed:
`c5dca9cc893ba55bede01f39211ceab1037738554340c82922871cbee2ea66d4`.
The original frozen parent and its runtime archives are preserved separately.
No GPU used, no remaining pilot job, no upload or submission-file replacement.

Next: extend collection to retain every observed pop as a separate event for
future route-reward attribution (without crediting unexecuted alternatives),
then bounded collection on additional training-pool seeds. Warm-start imitation can reproduce current
choices but cannot by itself establish better routing. Keep fresh paired
validation separate for any subsequently enabled learned selector. Do not
assign false hit labels to routes the rocket never flew.
