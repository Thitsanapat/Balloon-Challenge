# Scenario 4: observed-state reachable refresh (7 October 2026)

The qualification launcher and reviewed standalone agent are **unchanged**.
This is an observation/public-parameter-only development experiment, not a
submission or a claim of 20 pops. The official simulator, collision rule, and
evaluator were not changed. Offline metrics are never passed back to the agent.

## What was tested

`Scenario4ReachableRefreshAgent` starts with the selected guarded-innovation
planner. At a committed-leg update it screens the first 1.2 seconds of the
new spline against an optimistic cone from the *observed* body axis, last
throttle output, public axis/throttle rate limits, and remaining public motor
burn time. A large predicted shortfall can cause a joint re-solve of the whole
observed route with at most 1.2 seconds added to its first leg; only a densely
validated, materially easier candidate is accepted. If none is found, the
already-feasible nominal refresh is retained.

Separately, if the nominal committed refresh **already fails** the public
thrust/tilt/slew checks, it re-solves all arrival velocities and accelerations
at the **unchanged deadlines** and keeps every downstream target. This is the
previously promising full-route joint-refresh idea, now composed with the
selected guarded controller. It avoids waiting on a stale spline when a new
one from the current rocket estimate is feasible.

The preview is a ranking heuristic, not a 6-DOF reachable-set proof. It does
not know future gusts or true rocket/balloon positions.

## Complete official-simulator flights

| Scenario 4 seed | Selected standalone agent | Candidate | Candidate full-route rescues | Preview flags |
| ---: | ---: | ---: | ---: | ---: |
| 0 (development) | 5 | 6 | 4 | 0 |
| 1 (development) | 4 | 4 | 4 | 0 |
| 2 (development) | 3 | 4 | 4 | 0 |
| 1552368575 (fresh predeclared pair) | 7 | 7 | 4 | 0 |

All candidate flights completed normally with source dependencies unchanged
during each run. On seed 2 the candidate added target #12 to the three hits
from the selected agent. On the fresh seed the first five hits were shared,
but the final two target IDs differed: selected #74/#99, candidate #39/#50.
The fresh seed was drawn and recorded in
`qualification/used_seed_manual.jsonl` before either flight. The generated
`qualification/used_seeds.json` was rebuilt afterwards and now includes it;
older unlogged training seeds remain a registry coverage caveat.

The initial preview-only version (before adding joint refresh) scored 3 on
development seed 2, exactly tying the selected 3, and took no preview action.
In the final candidate, `reach_screened` was 0/1/1/0 on the four seeds, and
`reach_flagged` was zero throughout. Thus the observed gains of +1 on seeds
0 and 2 are attributable to the **joint derivative refresh**, not the new
short-horizon actuator screen. A low screen count is not evidence that
actuators are never limiting: the controller may encounter difficult tracking
outside its narrow committed-refresh window, and sensor/forecast error matters.

Reports (local, credential-free, not upload JSON):

- `evaluation/results/s4_reachable_refresh_dev_seed2_20261007.json`:
  preliminary preview-only score 3.
- `evaluation/results/s4_reachable_joint_dev_seed{0,1,2}_20261007.json`:
  final candidate development scores 6/4/4.
- `evaluation/results/s4_reachable_holdout_{baseline,candidate}_seed1552368575_20261007.json`:
  paired fresh scores 7/7.

The selected standalone source SHA-256 remains
`dab27623c797c70c6fc5c13b48e4faa2e146a694c7de737f6cae8138485fb333`.
The final candidate source SHA-256 in the official metrics is
`318128632d722975b1ec29e942e18494d5eb66cdc6b9c7b32b9ff593f570c3ab`.
Fourteen focused tests across reachable refresh, joint refresh, and guarded
innovation pass. The candidate is not yet a standalone review package.

## Decision

Do **not** replace the live submission: +2 pops across three familiar
development seeds and a tie on one fresh pair do not establish a robust gain
for independent live seeds. The safe next test is a predeclared multi-seed
paired holdout of the joint-refresh composition, separating it from the
preview screen. To make the preview useful, calibrate a near-term prediction
of *actual tracking error* from observations and actuator history, then
validate its error and intervention rate before changing flight commands.
