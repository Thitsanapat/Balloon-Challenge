# Scenario 4: seed-0 development checkpoint, 2026-10-04

Goal: work toward 10 actual pops on development seed 0, without embedding
seed-specific target IDs, prerecorded commands, or future balloon data.
This is a development benchmark, not a banked live-qualification score.

The selected multi-seed policy remains the 42-second wind-profile agent
(5 pops on seed 0). An older adaptive-feedback candidate recorded 6 pops on
seed 0 but lost its eight-seed paired comparison (30 versus 33 total pops),
so the single-seed record is not the same as the selected robust policy.

## Experiments

1. `Scenario4InterceptCandidatesAgent`: retain half the nearest-target
   candidates and screen the remainder over multiple interception horizons.
   The proxy includes observed target drift, rocket velocity, gravity,
   observation-estimated disturbance, public thrust/mass, and tilt limits.
   It only orders candidates; ordinary spline feasibility checks still run.
   The proxy does not model the finite time needed to change acceleration.
2. `Scenario4JointRefreshAgent`: when the existing committed-leg update fails,
   re-solve the remaining spline chain's arrival velocities and accelerations
   at unchanged deadlines. Check the complete proposed chain at 65 samples
   against the existing thrust-reserve, tilt, slew, throttle-rate, and ground
   constraints. If enabled, try shorter prefixes when the full chain fails.
   Keep the old plan unchanged when every attempt fails.

| Fresh official environment, seed 0 | Pops | Wall seconds | Decision |
| --- | ---: | ---: | --- |
| Established selected policy (prior run) | 5 | 315.3 | Keep |
| Multi-horizon candidate screening | 5 | 245.4 | No demonstrated score gain |
| Joint derivative refresh, shortening allowed | 2 | 298.8 | Reject |
| Joint derivative refresh, complete route only | 6 | 389.0 | +1 versus selected policy; development candidate only |

The candidate-screening flight terminated normally, was not truncated, and
all loaded agent source hashes remained unchanged during evaluation. It
changed candidate sets on 244 of 257 calls, so this was an active intervention,
not a disabled feature. Its first target remained the same as the baseline;
later route changes did not increase actual pops. Differences in wall time
are not a controlled runtime benchmark.

The joint refresh flight also terminated normally with unchanged source, but
lost three pops. It accepted six rescues and dropped eight planned future
legs cumulatively. Saturation rose to 1,399 steps (selected baseline: 748).
Its first hit was slightly earlier, which did not predict final score. A
controlled ablation disables prefix shortening without changing any other
setting, to separate route deletion from the derivative-refresh mechanism.

The complete-route-only ablation finished with **6 actual pops**, no truncation,
and unchanged source hashes. It kept all five baseline pops at identical
times and additionally popped target 48 at 69.67 s. It accepted four of 35
joint refresh attempts and dropped zero planned legs; saturation was 733
steps versus 748 for the baseline. This supports keeping the constrained
refresh for further development, not the permissive shortening variant.
This is a +1 result against the selected policy on one development seed,
ties the older experimental seed-0 record, and does **not** achieve 10 pops
or establish improvement on randomly drawn live seeds.

The exact successful kwargs are frozen in
`evaluation/configs/scenario4_joint_refresh_full_route_launch42.yaml`.
The actual receipt records the original experimental config with the explicit
CLI override `--set joint_refresh_allow_shortening=false`; these kwargs match
the dedicated config. The selected reviewed submission remains untouched.

Diagnostic caution: the closest-approach telemetry's tracking vector uses
raw GNSS, whose published position standard deviation is 2 m per coordinate.
Do not interpret its 2--3 m error as pure controller tracking error. The
controller's mean filtered-state reference error was 0.468 m for the selected
policy, 1.027 m for candidate screening, and 0.579 m for joint refresh with
shortening. Evaluator-only navigation RMS errors were 0.474/0.448/0.464 m.
These aggregate diagnostics do not by themselves explain individual misses
or substitute for actual pops, and simulator truth never enters either agent.

Credential-free metrics live in `BalloonPoppingGymEnv/evaluation/results/`:

- `s4_dev_intercept_candidates_seed0_20261004.json`
- `s4_dev_joint_refresh_seed0_20261004.json`
- `s4_dev_joint_refresh_full_route_seed0_20261004.json`

These are diagnostic reports, **not submission JSON files**. No submission was
uploaded, and no existing reviewed agent or official simulator component was
changed. Runs use the local CPU; no professor GPU was used.
Read-only `git diff --quiet v0.2.2 --` for the environment, official evaluator,
submission packer, and official verifier returned zero. The release reference
is `bc51609b24d0ff2335d8b35679a29132502ff8b3`; the audit script's older default
reference `bd9bb51` predates Scenario 4 and is not appropriate for this check.

All 111 Scenario-4 unit tests passed, including nine tests for the new
candidate-screening and joint-refresh mechanisms. These check time alignment,
momentum-aware screening, exclusions, fallback, endpoint prediction, dense
validation, and the explicit route-shortening switch. Unit tests do not
establish score improvement.

Rules reference: [organizer clarification #165](https://github.com/ARRC-Rocket/BalloonPoppingChallenge/discussions/165).
Only observations and given parameters enter these agents. The seed is set
externally by the evaluator. A future live seed is not known in advance.

## Follow-up: interaction test and cross-seed check

Keep the six-pop full-route-refresh source/config unchanged. Evaluate it on
development seed 1 against the established score of 2, rather than assuming
the seed-0 gain generalizes. Separately, `Scenario4JointInnovationAgent`
composes that rescue hook with the existing bounded live endpoint-innovation
controller. The latter compares the current observed forecast to the active
spline endpoint; it does not receive stored trajectories. This tests whether
remaining forecast error can be corrected without discarding the route.
No parameter sweep, seed lookup, or new model training is part of this test.

| Follow-up | Seed | Pops | Status |
| --- | ---: | ---: | --- |
| Frozen full-route refresh | 1 | 3 | Complete, +1 versus selected baseline |
| Full-route refresh + forecast innovation | 0 | 6 | Complete, ties full-route-only count |

Reports: `s4_dev_joint_refresh_full_route_seed1_20261004.json` and
`s4_dev_joint_innovation_seed0_20261004.json`. The new composition has unit
tests for initialization of both components, method dispatch, exact zero-
innovation-strength equivalence to the full-route controller, and observation
snapshot cleanup after exceptions. All 114 Scenario-4 unit tests passed.
Two local CPU evaluations run with BLAS/OpenMP thread counts limited to one
per process; no GPU is used. They do not change the reviewed submitted agent.

The frozen full-route refresh completed seed 1 in 330.7 s without truncation
or source changes. It accepted three of 59 rescue attempts and dropped no
legs. Together with seed 0, the development comparison is **6/3 versus 5/2**
for the selected baseline. Both are known development seeds, not a new held-
out validation set. This is encouraging but insufficient to promote a policy
for random live conditions, and the total of 9 is not a submitted/live score.

The combined controller completed seed 0 in 321.9 s, with unchanged source
and no truncation. It applied forecast corrections on 2,164 steps, accepted
four of 39 joint refresh attempts, and dropped no legs. Its six actual pops
ended at 69.44 s versus 69.67 s for full-route refresh alone. The later target
sequence differs, so this is not merely the same route flown faster. The
count ties and the 0.23-second last-hit improvement are exploratory; keep it
as an unpromoted candidate until the extra feedback proves useful on paired
seeds. It has not reached 10 pops. No evaluation jobs remain running after
these two completed flights; no new submission JSON was packaged or uploaded.

## Why Scenarios 2 and 3 exist

Scenario numbers describe simulation conditions, not a compulsory sequence of
scoring rounds. The organizer's September 18 email explicitly offered 2 and 3
for training ahead of the qualification parameters. The published README's
scenario table describes 2 as noisy sensing, 3 as noisy sensing plus actuator
dynamics, and 4 as these issues plus gusts. In the local official v0.2.2
parameters, 2 and 3 differ in the three actuator time constants; 3 and 4 differ
in the gust settings. This makes them useful diagnostic/curriculum cases:
separate estimation problems from actuator lag and atmospheric forecast error.
That is a development use, not an extra qualification score.
Do not assume the same seed yields the same balloon field across scenarios:
enabling gusts changes random-number use and the atmosphere. Cross-scenario
score differences alone do not isolate a controller defect.

The current qualification rules specify Scenario 4, two live flights with new
seeds, and the sum of their pop counts. Local seed-0 results and scores from
2 or 3 cannot be banked into that total. Prioritize Scenario 4 for selection;
use 2/3 only when a focused diagnostic or training experiment justifies it.
The README's general scenario table retains some TBD labels; the released
qualification parameters and explicit competition rules are the relevant
references, not an assumption that every numbered scenario is a round.

Sources: [official competition rules and scenario descriptions](https://github.com/ARRC-Rocket/BalloonPoppingChallenge/blob/main/README.md),
[v0.2.2 release](https://github.com/ARRC-Rocket/BalloonPoppingChallenge/releases/tag/v0.2.2),
and the organizer email supplied by the user in this conversation.
